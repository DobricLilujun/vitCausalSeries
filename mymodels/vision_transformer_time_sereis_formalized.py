import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils.spt import ShiftedPatchTokenization
from torch import einsum
from einops import rearrange, repeat


class PatchDecodeTimeSeries(nn.Module):
    def __init__(self, patch_size, embed_dim, out_chans, time_span=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.time_span = time_span
        self.out_chans = out_chans
        self.conv_output_channel = out_chans // time_span

        # Define the transpose convolution layer
        self.deproj = nn.ConvTranspose2d(
            embed_dim,
            self.conv_output_channel,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.apply(self._init_weights)

    def forward(self, x):
        B, T, N, P = (
            x.shape
        )  # T is the concatenated time steps, N is the patch count per frame
        assert (
            T % self.time_span == 0
        ), "Total sequence dimension must be divisible by the number of time steps (T)."

        # Separate time steps
        frame_patches = torch.split(
            x, split_size_or_sections=T // self.time_span, dim=1
        )

        # Initialize a list to store decoded frames
        decoded_frames = []

        for t, frame in enumerate(frame_patches):
            # Reshape to (B, embed_dim, H', W') for transpose convolution
            patch_size_sqrt = int(P**0.5)
            frame = frame.view(B, self.embed_dim, patch_size_sqrt, patch_size_sqrt)

            # Apply transpose convolution to decode the frame
            decoded_frame = self.deproj(frame)

            decoded_frames.append(decoded_frame)

        # Concatenate decoded frames along the time dimension
        x = torch.stack(decoded_frames, dim=1)  # (B, T, C', H, W)

        # Combine time and channel dimensions back to original shape
        x = x.view(B, self.time_span * self.conv_output_channel, *x.shape[3:])

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features
        hidden_features=None,  # Number of hidden features, defaults to input features if not provided
        out_features=None,  # Number of output features, defaults to input features if not provided
        act_layer=nn.GELU,  # Activation function, default is GELU
        drop=0.0,  # Dropout rate, default is 0.0 (no dropout)
    ):
        super().__init__()
        # Set the output and hidden features, defaulting to input features if not provided
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define the first fully connected layer (fc1)
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Define the activation layer (default is GELU)
        self.act = act_layer()
        # Define the second fully connected layer (fc2)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Define the dropout layer
        self.drop = nn.Dropout(drop)

        # Initialize weights using the custom weight initialization method
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Custom weight initialization
        if isinstance(m, nn.Linear):
            # Initialize weights with truncated normal distribution
            trunc_normal_(m.weight, std=0.02)
            # Initialize biases to zero
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm weights and biases
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Forward pass through the layers
        x = self.fc1(x)  # First fully connected layer
        x = self.act(x)  # Activation function
        x = self.drop(x)  # Dropout layer
        x = self.fc2(x)  # Second fully connected layer
        x = self.drop(x)  # Dropout layer
        return x


class GPSA(nn.Module):
    def __init__(
        self,
        dim,  # Dimensionality of input/output features
        num_heads=8,  # Number of attention heads
        qkv_bias=False,  # Whether to include a bias in the qkv projections
        qk_scale=None,  # Scaling factor for qk dot product
        attn_drop=0.0,  # Dropout rate for attention
        proj_drop=0.0,  # Dropout rate for output projection
        locality_strength=1.0,  # Strength of locality-based attention
        use_local_init=True,  # Whether to initialize the locality parameters
        grid_size=None,  # Size of the grid (height, width) for relative attention
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads  # Calculate head dimension
        self.scale = qk_scale or head_dim**-0.5  # Scaling factor for qk dot product

        # Define linear projections for query, key, and value
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # Dropout and projection layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(
            3, num_heads
        )  # Positional projection for relative attention
        self.proj_drop = nn.Dropout(proj_drop)

        # Locality and gating parameters
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(
            1 * torch.ones(self.num_heads)
        )  # Gating parameter for blending attention types
        self.apply(self._init_weights)  # Initialize weights using custom method

        # Local initialization if enabled
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

        # Grid size for attention (used in relative positioning)
        self.current_grid_size = grid_size

    def _init_weights(self, m):
        # Custom weight initialization for linear layers and LayerNorm
        if isinstance(m, nn.Linear):
            trunc_normal_(
                m.weight, std=0.02
            )  # Initialize weights with truncated normal distribution
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to 0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # Initialize LayerNorm bias to 0
            nn.init.constant_(m.weight, 1.0)  # Initialize LayerNorm weight to 1.0

    def get_attention(self, x):
        B, N, C = x.shape  # Batch size, number of tokens, number of channels (features)

        # Compute the key, query, and value projections
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # Compute positional score (relative attention)
        pos_score = (
            self.pos_proj(self.rel_indices).expand(B, -1, -1, -1).permute(0, 3, 1, 2)
        )

        # Compute patch scores (dot product attention between queries and keys)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(
            dim=-1
        )  # Apply softmax for attention distribution
        pos_score = pos_score.softmax(dim=-1)  # Normalize positional scores

        # Compute the gating mechanism (blending patch-based and positional attention)
        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1.0 - torch.sigmoid(gating)) * patch_score + torch.sigmoid(
            gating
        ) * pos_score
        attn = attn / attn.sum(dim=-1).unsqueeze(-1)  # Normalize attention scores
        attn = self.attn_drop(attn)  # Apply attention dropout
        return attn

    def get_attention_map(self, x, return_map=False):
        # Compute the attention map by averaging over the batch
        attn_map = self.get_attention(x).mean(0)  # Average attention over batch
        distances = (
            self.rel_indices.squeeze()[:, :, -1] ** 0.5
        )  # Compute the distance for relative positions
        dist = torch.einsum(
            "nm,hnm->h", (distances, attn_map)
        )  # Compute distance-weighted attention
        dist /= distances.size(0)  # Normalize distances
        if return_map:
            return dist, attn_map  # Return both distance and attention map if requested
        else:
            return dist  # Return just the distance

    def local_init(self, locality_strength=1.0):
        # Initialize the locality parameters (relative positioning) for attention
        self.v.weight.data.copy_(
            torch.eye(self.dim)
        )  # Initialize value projections with identity matrix
        locality_distance = 1  # Maximum locality distance

        # Define kernel size for the locality grid
        kernel_size = int(self.num_heads**0.5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2

        # Fill the position projection weights based on locality
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = (
                    -1
                )  # Set locality distance in position projections
                self.pos_proj.weight.data[position, 1] = (
                    2 * (h1 - center) * locality_distance
                )
                self.pos_proj.weight.data[position, 0] = (
                    2 * (h2 - center) * locality_distance
                )
        self.pos_proj.weight.data *= locality_strength  # Scale the locality parameters

    def get_rel_indices(self):
        # Compute relative position indices for the attention grid
        H, W = self.current_grid_size
        N = H * W
        rel_indices = torch.zeros(1, N, N, 3)
        indx = torch.arange(W).view(1, -1) - torch.arange(W).view(-1, 1)
        indx = indx.repeat(H, H)
        indy = torch.arange(H).view(1, -1) - torch.arange(H).view(-1, 1)
        indy = indy.repeat_interleave(W, dim=0).repeat_interleave(W, dim=1)
        indd = indx**2 + indy**2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.v.weight.device
        self.rel_indices = rel_indices.to(
            device
        )  # Store relative indices on the appropriate device

    def forward(self, x):
        B, N, C = x.shape  # Batch size, number of tokens, number of channels (features)
        if not hasattr(self, "rel_indices") or self.rel_indices.size(1) != N:
            self.get_rel_indices()  # Update relative indices if necessary

        attn = self.get_attention(x)  # Compute attention
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        # Apply attention to values and project the output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # Output projection
        x = self.proj_drop(x)  # Apply projection dropout
        return x


def init_weights(m):
    # Initialize weights using Xavier normal distribution
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def exists(val):
    # Check if a value is not None
    return val is not None


def rotate_every_two(x):
    # Rotate the tensor for positional encoding
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(q, k, sinu_pos):
    # Apply rotary positional embedding to queries and keys
    sinu_pos = rearrange(sinu_pos, "() n (j d) -> n j d", j=2)
    sin, cos = sinu_pos.unbind(dim=-2)

    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


class MHSA(nn.Module):
    def __init__(
        self,
        dim,
        num_patches,
        heads=8,
        dim_head=64,
        time_span=6,
        dropout=0.0,
        is_LSA=False,
        is_CSA=False,
    ):
        super().__init__()
        inner_dim = (
            dim_head * heads
        )  # The dimension after combining all heads in attention
        project_out = not (
            heads == 1 and dim_head == dim
        )  # Check if output projection is needed
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head**-0.5  # Scaling factor for dot-product attention
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)  # Softmax for attention weights
        self.to_qkv = nn.Linear(
            self.dim, self.inner_dim * 3, bias=False
        )  # Linear transformation for Q, K, V
        init_weights(self.to_qkv)  # Initialize weights
        self.to_out = (
            nn.Sequential(
                nn.Linear(self.inner_dim, self.dim), nn.Dropout(dropout)
            )  # Project back to input dimension
            if project_out
            else nn.Identity()  # If no projection needed, just return the input
        )
        self.time_span = time_span  # Time steps for causal attention

        # If Local Self Attention (LSA) is enabled, create local masks
        if is_LSA:  ## Local Self Attention
            self.scale = nn.Parameter(
                self.scale * torch.ones(heads)
            )  # Learnable scaling factor for each head
            self.mask = torch.eye(
                self.num_patches + 1, self.num_patches + 1
            )  # Initialize identity mask
            self.mask = torch.nonzero(
                (self.mask == 1), as_tuple=False
            )  # Get non-zero positions (identity mask)

        ############### This is the part that needs to be changed #################
        # elif is_CSA:  ## Causal Self Attention (commented-out section)
        #     time_span = self.time_span
        #     self.scale = nn.Parameter(self.scale * torch.ones(heads))  # Learnable scaling factor for each head
        #     self.mask = torch.zeros(
        #         self.num_patches,
        #         self.num_patches,
        #     )  # Initialize mask as zeros
        #     num_patches_per_time = self.num_patches // time_span
        #     # Apply spatial attention
        #     for t in range(time_span):
        #         cur_time_start_idx = t * num_patches_per_time
        #         cur_time_end_idx = cur_time_start_idx + num_patches_per_time
        #         self.mask[
        #             cur_time_start_idx:cur_time_end_idx,
        #             cur_time_start_idx:cur_time_end_idx,
        #         ] = 1  # Mask diagonal blocks
        #     # Add causal relationships between time steps
        #     for t in range(1, time_span):
        #         start_idx = t * num_patches_per_time
        #         prev_idx = (t - 1) * num_patches_per_time
        #         self.mask[
        #             start_idx : start_idx + num_patches_per_time,
        #             prev_idx : prev_idx + num_patches_per_time,
        #         ] = 1  # Add attention to previous time step
        #         next_idx = (t + 1) * num_patches_per_time
        #         if t < time_span - 1:
        #             self.mask[
        #                 start_idx : start_idx + num_patches_per_time,
        #                 next_idx : next_idx + num_patches_per_time,
        #             ] = 1  # Add attention to next time step
        #     self.mask = torch.ones(
        #         self.num_patches,
        #         self.num_patches,
        #     )  # Only for testing
        #     self.mask = torch.nonzero((self.mask == 1), as_tuple=False)  # Get non-zero positions

        else:
            self.mask = None  # No mask for standard self-attention

    def forward(self, x, pos_emb=None):
        b, n, _, h = (
            *x.shape,
            self.heads,
        )  # Batch size, number of patches, channels, number of heads
        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # Split input into Q, K, V (queries, keys, values)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv
        )  # Rearrange for multi-head attention

        if exists(pos_emb):
            q, k = apply_rotary_pos_emb(
                q, k, pos_emb
            )  # Apply rotary positional encoding if provided

        if self.mask is None:
            dots = (
                einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
            )  # Standard attention (dot-product)

        else:
            scale = self.scale  # Scaling factor
            dots = torch.mul(
                einsum("b h i d, b h j d -> b h i j", q, k),  # Dot-product attention
                scale.unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand((b, h, 1, 1)),  # Apply scale per head
            )
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = (
                -987654321
            )  # Apply mask by setting invalid positions to a very large negative value

        attn = self.attend(dots)  # Apply softmax to compute attention weights
        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # Apply attention to values

        out = rearrange(
            out, "b h n d -> b n (h d)"
        )  # Rearrange output to combine attention heads
        return self.to_out(out)  # Project output back to the original input dimension


# This Part Of Block stay unchanged from the original code for Multi-block Transformer Structure
# This part of the code defines the Transformer block structure, which remains unchanged in the multi-block Transformer model.


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_gpsa=True,
        num_patches=None,
        is_LSA=False,
        is_CSA=False,
        time_span=6,
        **kwargs
    ):
        super().__init__()

        # Layer normalization for the first attention layer, conditionally using the CSA flag
        if not is_CSA:
            self.norm1 = norm_layer(dim)
        else:
            self.norm1 = norm_layer(dim)

        self.use_gpsa = use_gpsa

        # Use Global Patch Self-Attention (GPSA) if specified
        if self.use_gpsa:
            self.attn = GPSA(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                **kwargs
            )
        else:
            # Otherwise, use Multi-Head Self-Attention (MHSA) for attention mechanism
            # self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches, is_LSA=is_LSA, **kwargs)
            self.attn = MHSA(
                dim,
                num_patches,
                heads=num_heads,
                dim_head=44,  # Dimensionality of each head
                dropout=drop,
                is_LSA=False,
                is_CSA=is_CSA,  # Whether to use causal self-attention
            )
            # Define another MHSA for temporal attention
            self.temp_attn = MHSA(
                dim,
                num_patches,
                heads=num_heads,
                dim_head=44,
                dropout=drop,
                is_LSA=False,
                is_CSA=is_CSA,
            )

        # DropPath for regularization, if specified
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer normalization for the MLP layer
        self.norm2 = norm_layer(dim)

        # Define the MLP (multi-layer perceptron) part of the block
        mlp_hidden_dim = int(dim * mlp_ratio)  # Hidden dimension of MLP
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Time span for causal self-attention
        self.time_span = time_span
        self.is_CSA = is_CSA  # Flag to determine if causal self-attention is used

    def forward(self, x, grid_size, pos_emb=None):
        # Set the current grid size for attention
        self.attn.current_grid_size = grid_size

        # Apply the attention mechanism
        pos_attention = self.drop_path(self.attn(self.norm1(x), pos_emb=pos_emb))

        if self.is_CSA:
            # Reshape for temporal attention if using causal self-attention
            B_T, seq_len, embed_dim = x.shape
            H_p = int(seq_len**0.5)  # Height of the grid
            W_p = H_p  # Width of the grid
            B = B_T // self.time_span  # Batch size for time steps

            # Reshape and permute the input for temporal attention processing
            x_temporal = x.view(B, self.time_span, H_p * W_p, embed_dim)  # [B, T, S, E]
            x_temporal = x_temporal.permute(
                0, 2, 1, 3
            )  # Permute dimensions for temporal attention
            x_temporal = x_temporal.reshape(
                H_p * W_p * B, self.time_span, embed_dim
            )  # [B * S, T, E]

            # Apply temporal attention
            temp_attention = self.drop_path(
                self.temp_attn(self.norm1(x_temporal), pos_emb=pos_emb)
            )

            # Rearrange the output of the temporal attention
            temp_attention = rearrange(
                temp_attention, "(b s) t e -> b s t e", b=B, s=H_p * W_p
            )
            temp_attention = rearrange(temp_attention, "b s t e -> b t s e")
            temp_attention = rearrange(temp_attention, "b t s e -> (b t) s e")

            # Add the temporal attention output back to the input
            x = x + temp_attention
            # x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Apply the positional attention and MLP layers
        x = x + pos_attention
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbedTimeSeries(nn.Module):
    """
    Time-Series Patch Embedding for multichannel input where temporal dimension is embedded as separate time slices.
    Converts the input image sequence into patch embeddings using a Conv2D layer.
    """

    def __init__(self, patch_size, in_chans, embed_dim, time_span=3):
        super().__init__()
        # Number of channels per time step
        self.conv_input_channel = in_chans // time_span
        self.time_span = time_span

        # Convolution to extract non-overlapping patches from each time slice
        self.proj = nn.Conv2d(
            self.conv_input_channel,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # Apply weight initialization
        self.apply(self._init_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure that the number of channels is divisible by the number of time steps
        assert (
            C % self.time_span == 0
        ), "Total channels must be divisible by the number of time steps (T)."

        # Reshape input: split temporal channels into separate time steps
        x = x.view(B, self.time_span, self.conv_input_channel, H, W)

        # Merge batch and time dimension to apply convolution to each time step
        x = x.view(B * self.time_span, self.conv_input_channel, H, W)

        # Extract patch embeddings
        x = self.proj(x)

        return x

    def _init_weights(self, m):
        # Initialize weights for Linear and LayerNorm layers
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding, adapted from the timm library"""

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        # Use a Conv2D layer to extract non-overlapping patches from the image
        # The kernel size and stride are both set to patch_size to ensure no overlap
        # Output shape: [B, embed_dim, H/patch_size, W/patch_size]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        # Apply custom weight initialization to the module
        self.apply(self._init_weights)

    def forward(self, x):
        # Project the input image into patch embeddings
        x = self.proj(x)
        return x

    def _init_weights(self, m):
        # Custom weight initialization function
        if isinstance(m, nn.Linear):
            # Initialize linear weights with truncated normal distribution
            trunc_normal_(m.weight, std=0.02)
            # Initialize bias to zero if present
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # For LayerNorm, initialize bias to 0 and weight to 1
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()

        # Compute the inverse frequency for each dimension (even indices)
        # Based on the formula used in sinusoidal positional encodings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        # Generate position indices from 0 to max_seq_len - 1
        position = torch.arange(0, max_seq_len, dtype=torch.float)

        # Compute the outer product of position and inverse frequency
        # Resulting shape: [max_seq_len, dim // 2]
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)

        # Apply sine and cosine to alternate dimensions and concatenate
        # Final embedding shape: [max_seq_len, dim]
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

        # Register as a buffer so it's saved with the model but not updated during training
        self.register_buffer("emb", emb)

    def forward(self, x):
        # Return positional embeddings sliced to match the input sequence length
        # Shape returned: [1, sequence_length, dim]
        return self.emb[None, : x.shape[1], :].to(x.device)


# Predefined Structure Of Time Series Vision Transformer
class VisionTransformerTimeSeriesFormalized(nn.Module):
    def __init__(
        self,
        avrg_img_size=320,  # Average image size used to compute spatial dimensions
        patch_size=10,  # Size of each patch
        in_chans=1,  # Number of input channels (e.g., for grayscale = 1)
        embed_dim=64,  # Embedding dimension per head
        depth=8,  # Number of transformer blocks
        num_heads=9,  # Number of attention heads
        mlp_ratio=4.0,  # Ratio of hidden size to embedding size in MLP
        qkv_bias=False,  # Whether to use bias in QKV projections
        qk_scale=None,  # Optional QK scale
        drop_rate=0.0,  # Dropout rate
        attn_drop_rate=0.0,  # Attention dropout rate
        drop_path_rate=0.0,  # Stochastic depth rate
        norm_layer=nn.LayerNorm,  # Normalization layer
        global_pool=None,  # Optional global pooling (not used here)
        gpsa_interval=[
            -1,
            -1,
        ],  # Interval of blocks using GPSA (Global Position Self-Attention)
        locality_strength=1.0,  # GPSA locality strength
        use_pos_embed=True,  # Whether to use spatial positional embeddings
        use_time_embed=False,  # Whether to use temporal embeddings
        is_LSA=False,  # Whether to use Local Self-Attention
        is_SPT=False,  # Whether to use Shifted Patch Tokenization
        is_CSA=False,  # Whether to use Channel Self-Attention (for time-series)
        rotary_position_emb=False,  # Whether to use rotary positional embeddings
    ):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        embed_dim *= num_heads  # Total embedding dimension = per-head dim * number of heads  # Set the embedding dimension for the model
        self.num_features = embed_dim  # Set for compatibility with other models
        self.locality_strength = (
            locality_strength  # Locality strength for GPSA but not used in our paper
        )
        self.use_pos_embed = use_pos_embed  # Whether to use spatial positional embeddings which is important for time-series image dataset
        self.use_time_embed = use_time_embed  # Whether to use temporal embeddings which is not used in our paper's main experiments
        self.is_CSA = is_CSA  # Whether to use causal self attention

        # Convert image size and patch size to tuples if given as int
        if isinstance(avrg_img_size, int):
            img_size = to_2tuple(
                avrg_img_size
            )  # Convert to tuple: For example: (180, 180)
        if isinstance(patch_size, int):
            self.patch_size = to_2tuple(
                patch_size
            )  # Convert to tuple: For example:  (5, 5)
        else:
            self.patch_size = patch_size

        self.in_chans = in_chans  # Number of input channels  = Original image channels number * time span = 11 * 6 For example

        # Module to decode patches back to time series (used later in output) Not used in our paper
        # self.PatchDeodceTimeSeries = PatchDecodeTimeSeries(
        #     patch_size, in_chans, embed_dim, time_span=6
        # )

        # Patch embedding layer selection based on SPT and CSA flags
        if not is_SPT:
            if not is_CSA:
                # Standard patch embedding using Conv2D
                self.patch_embed = PatchEmbed(
                    patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim
                )
            else:
                self.patch_embed = PatchEmbedTimeSeries(
                    patch_size=self.patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    time_span=in_chans // 11,
                )
        else:
            self.patch_embed = ShiftedPatchTokenization(
                in_chans, embed_dim, merging_size=self.patch_size, is_pe=True
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Learnable spatial positional embedding (used if enabled)
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    embed_dim,
                    img_size[0] // self.patch_size[0],
                    img_size[1] // self.patch_size[1],
                )
            )
            trunc_normal_(self.pos_embed, std=0.02)

        # Learnable temporal embedding (if enabled) Not used in our paper
        # if self.use_time_embed:
        #     self.time_embed = nn.Parameter(
        #         torch.zeros(
        #             1,
        #             embed_dim,
        #             self.in_chans // 11,
        #         )
        #     )

        # Compute number of patches
        if not is_CSA:
            num_patches = (img_size[0] // self.patch_size[0]) * (
                img_size[1] // self.patch_size[1]
            )
        else:
            num_patches = (
                (img_size[0] // self.patch_size[0])
                * (img_size[1] // self.patch_size[1])
                * (self.in_chans // 11)
            )

        # Create drop path schedule for each transformer block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Create transformer blocks
        self.blocks = nn.ModuleList(
            [
                (
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        use_gpsa=True,
                        locality_strength=locality_strength,
                        num_patches=num_patches,
                        is_LSA=is_LSA,
                        is_CSA=self.is_CSA,
                    )
                    if gpsa_interval[0] - 1 <= i < gpsa_interval[1]
                    else Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        use_gpsa=False,
                        num_patches=num_patches,
                        is_LSA=is_LSA,
                        is_CSA=self.is_CSA,
                    )
                )
                for i in range(depth)
            ]
        )

        # Final layer normalization
        self.norm = norm_layer(embed_dim)

        # Define output head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        if not self.is_CSA:
            self.head = nn.Linear(
                self.num_features, in_chans * self.patch_size[0] * self.patch_size[1]
            )
        else:
            self.head = nn.Linear(
                self.num_features, 11 * self.patch_size[0] * self.patch_size[1]
            )

        # Positional embedding for each transformer layer (if rotary encoding is used)
        max_seq_len = 1025
        self.layer_pos_emb = None
        if rotary_position_emb:
            self.layer_pos_emb = FixedPositionalEmbedding(self.embed_dim, max_seq_len)

    def seq2img(self, x, img_size):
        """
        feature2img:
        Transforms sequence back into image space, input dims: [batch_size, num_patches, channels]
        output dims: [batch_size, channels, H, W]
        """
        if not self.is_CSA:
            # Reshape the input tensor to split patches into their individual spatial dimensions
            x = x.view(
                x.shape[0],
                x.shape[1],
                self.in_chans,
                self.patch_size[0],
                self.patch_size[1],
            )

            # Chunk the patches along the second dimension (num_patches) and concatenate along the spatial dimension
            x = x.chunk(x.shape[1], dim=1)
            x = torch.cat(x, dim=4).permute(0, 1, 2, 4, 3)

            # Further chunk the tensor along the height (H) dimension and concatenate along the spatial dimension
            x = x.chunk(img_size[0] // self.patch_size[0], dim=3)
            x = torch.cat(x, dim=4).permute(0, 1, 2, 4, 3).squeeze(1)
        else:
            # For CSA Causal Self Attention
            x = x.view(
                x.shape[0],
                x.shape[1],
                11,
                self.patch_size[0],
                self.patch_size[1],
            )

            # Get dimensions of the reshaped tensor
            batch_size, num_patches, num_channels, patch_size, _ = x.shape

            # Calculate the number of patches in width and height based on the image size
            patch_per_width = img_size[0] // self.patch_size[0]
            patch_per_height = patch_per_width

            # Calculate the temporal span (e.g., time steps)
            time_span = self.in_chans // 11

            # Reshape to combine patches in spatial dimensions (height and width)
            x = x.view(
                batch_size // time_span,
                time_span,
                num_channels,
                patch_per_width * patch_size,
                patch_per_height * patch_size,
            )

            # Final reshape to reconstruct the image by combining temporal and spatial dimensions
            x = x.view(
                batch_size // time_span,
                num_channels * time_span,
                patch_per_width * patch_size,
                patch_per_height * patch_size,
            )

        return x

    def _init_weights(self, m):
        # Check if the module is an instance of nn.Linear (a linear layer)
        if isinstance(m, nn.Linear):
            # Initialize the weights using truncated normal distribution with standard deviation of 0.02
            trunc_normal_(m.weight, std=0.02)

            # Check if the linear layer has a bias term and if so, initialize it to zero
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Check if the module is an instance of nn.LayerNorm (a layer normalization)
        elif isinstance(m, nn.LayerNorm):
            # Initialize the bias term of LayerNorm to zero
            nn.init.constant_(m.bias, 0)

            # Initialize the weight term of LayerNorm to 1.0
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(
        self,
    ):
        return {"pos_embed"}

    def get_head(
        self,
    ):
        return self.head

    def reset_head(
        self,
    ):
        self.head = nn.Linear(
            self.num_features, self.in_chans * self.patch_size[0] * self.patch_size[1]
        )

    def forward_features(self, x):
        # Get the batch size (B), number of channels (C), height (H), and width (W) of the input tensor
        B, C, H, W = x.shape

        # Apply patch embedding to the input tensor
        x = self.patch_embed(x)  # (B_T, conv_input_channel, H, W)

        # Check if positional embedding is to be used
        if self.use_pos_embed:
            if not self.is_CSA:
                # For non-CSA cases, get the dimensions of the embedded tensor
                _, _, H, W = x.shape  # B, E_dim, H, W

                # Interpolate the positional embedding to match the spatial size of the input
                pos_embed = F.interpolate(
                    self.pos_embed, size=[H, W], mode="bilinear", align_corners=False
                )

                # Add the positional embedding to the input tensor
                x = x + pos_embed
            else:
                #### To be fixed  ####
                # For CSA (possibly temporal dimension), get the dimensions of the tensor
                B_T, E_dim, H_p, W_p = x.shape
                T = B_T // B  # Determine the time steps (T)

                # Repeat the positional embedding across the batch and time dimension
                pos_embed = self.pos_embed.repeat(
                    B_T, 1, 1, 1
                )  # (B_T, E_dim, H_p, W_p)

                # Add the repeated positional embedding to the input tensor
                x = x + pos_embed

                # Optional code for adding temporal embeddings (currently commented out)
                # if self.use_time_embed:
                #     time_embed = self.time_embed.expand(B, -1, -1)
                #     time_embed_expanded = time_embed.unsqueeze(-1).unsqueeze(-1)
                #     time_embed_expanded = time_embed_expanded.expand(-1, -1, -1, W, W)
                #     time_embed_expanded = time_embed_expanded.reshape(B, E_dim, T_H, W)
                #     x = x + time_embed_expanded

        # Flatten the input tensor and transpose it for further processing
        x = x.flatten(2).transpose(1, 2)  # (B_T, seq_len, E_dim)

        # Apply dropout to the positional embeddings
        x = self.pos_drop(x)
        # Apply layer-specific positional embedding if available
        if self.layer_pos_emb is not None:
            layer_pos_emb = self.layer_pos_emb(x)
        else:
            layer_pos_emb = None

        # Pass the tensor through each block in the model
        for u, blk in enumerate(self.blocks):
            # Apply the block with positional embeddings (if available)
            x = blk(x, (W, W), pos_emb=layer_pos_emb)

        # Apply the final normalization layer
        x = self.norm(x)
        # Return the processed tensor
        return x

    def forward(self, x):
        # Get the batch size (B), number of channels (C), height (H), and width (W) of the input tensor
        _, _, H, W = x.shape  # B, C, H, W

        # Pass the input tensor through the feature extraction process (patch embedding, positional embedding, etc.)
        x = self.forward_features(x)  # B, N_patches, E_dim

        # Pass the features through the head (e.g., a classifier or a linear layer)
        x = self.head(x)  # B, N_patches, Original_channels * Patch_size * Patch_size

        # Convert the sequence back to image-like format (reconstruct the image from patch sequence)
        x = self.seq2img(
            x, (H, W)
        )  # Reconstruct the image to original height and width

        # Return the reconstructed image
        return x
