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
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GPSA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        locality_strength=1.0,
        use_local_init=True,
        grid_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(1 * torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)
        self.current_grid_size = grid_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention(self, x):
        B, N, C = x.shape

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

        pos_score = (
            self.pos_proj(self.rel_indices).expand(B, -1, -1, -1).permute(0, 3, 1, 2)
        )
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1.0 - torch.sigmoid(gating)) * patch_score + torch.sigmoid(
            gating
        ) * pos_score
        attn = attn / attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):

        attn_map = self.get_attention(x).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** 0.5
        dist = torch.einsum("nm,hnm->h", (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self, locality_strength=1.0):

        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads**0.5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = (
                    2 * (h1 - center) * locality_distance
                )
                self.pos_proj.weight.data[position, 0] = (
                    2 * (h2 - center) * locality_distance
                )
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(
        self,
    ):
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
        self.rel_indices = rel_indices.to(device)

    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, "rel_indices") or self.rel_indices.size(1) != N:
            self.get_rel_indices()

        attn = self.get_attention(x)
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def exists(val):
    return val is not None


def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(q, k, sinu_pos):
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
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        init_weights(self.to_qkv)
        self.to_out = (
            nn.Sequential(nn.Linear(self.inner_dim, self.dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.time_span = time_span

        if is_LSA:  ## Local Self Attention
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        # elif is_CSA:  ## Causal Self Attention
        #     time_span = self.time_span
        #     self.scale = nn.Parameter(self.scale * torch.ones(heads))
        #     self.mask = torch.zeros(
        #         self.num_patches,
        #         self.num_patches,
        #     )
        #     num_patches_per_time = self.num_patches // time_span
        #     # Spatial Attention
        #     for t in range(time_span):
        #         cur_time_start_idx = t * num_patches_per_time
        #         cur_time_end_idx = cur_time_start_idx + num_patches_per_time
        #         self.mask[
        #             cur_time_start_idx:cur_time_end_idx,
        #             cur_time_start_idx:cur_time_end_idx,
        #         ] = 1

        #     # 填充mask：时间步之间的空间注意力（相邻的时间步之间有注意力）

        #     for t in range(1, time_span):
        #         start_idx = t * num_patches_per_time
        #         prev_idx = (t - 1) * num_patches_per_time
        #         self.mask[
        #             start_idx : start_idx + num_patches_per_time,
        #             prev_idx : prev_idx + num_patches_per_time,
        #         ] = 1  # 向前相邻的时间步

        #         next_idx = (t + 1) * num_patches_per_time
        #         if t < time_span - 1:
        #             self.mask[
        #                 start_idx : start_idx + num_patches_per_time,
        #                 next_idx : next_idx + num_patches_per_time,
        #             ] = 1  # 向后相邻的时间步

        #             # self.mask[prev_time_start_idx:prev_time_end_idx, cur_time_start_idx:cur_time_end_idx] = 1
        #     self.mask = torch.ones(
        #         self.num_patches,
        #         self.num_patches,
        #     )  # only for test
        #     self.mask = torch.nonzero((self.mask == 1), as_tuple=False)

        else:
            self.mask = None

    def forward(self, x, pos_emb=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        if exists(pos_emb):
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        if self.mask is None:
            dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(
                einsum("b h i d, b h j d -> b h i j", q, k),
                scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)),
            )
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


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
        if not is_CSA:
            self.norm1 = norm_layer(dim)
        else:
            self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
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
            # self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches, is_LSA=is_LSA, **kwargs)
            self.attn = MHSA(
                dim,
                num_patches,
                heads=num_heads,
                dim_head=44,
                dropout=drop,
                is_LSA=False,
                is_CSA=is_CSA,
            )
            self.temp_attn = MHSA(
                dim,
                num_patches,
                heads=num_heads,
                dim_head=44,
                dropout=drop,
                is_LSA=False,
                is_CSA=is_CSA,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.time_span = time_span
        self.is_CSA = is_CSA

    def forward(self, x, grid_size, pos_emb=None):
        self.attn.current_grid_size = grid_size
        pos_attention = self.drop_path(self.attn(self.norm1(x), pos_emb=pos_emb))

        if self.is_CSA:
            B_T, seq_len, embed_dim = x.shape
            H_p = int(seq_len**0.5)
            W_p = H_p
            B = B_T // self.time_span
            # Reshape for temporal attention
            x_temporal = x
            x_temporal = x_temporal.view(
                B, self.time_span, H_p * W_p, embed_dim
            )  # [B, T, S, E]
            x_temporal = x_temporal.permute(0, 2, 1, 3)
            x_temporal = x_temporal.reshape(
                H_p * W_p * B, self.time_span, embed_dim
            )  # [B, S, T, E] ->  # [B * S, T, E]
            temp_attention = self.drop_path(
                self.temp_attn(self.norm1(x_temporal), pos_emb=pos_emb)
            )  # [B * S, T, E]
            temp_attention = rearrange(
                temp_attention, "(b s) t e -> b s t e", b=B, s=H_p * W_p
            )
            temp_attention = rearrange(temp_attention, "b s t e -> b t s e")
            temp_attention = rearrange(temp_attention, "b t s e -> (b t) s e")
            x = x + temp_attention
            # x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x + pos_attention
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedTimeSeries(nn.Module):

    def __init__(self, patch_size, in_chans, embed_dim, time_span=6):
        super().__init__()
        self.conv_input_channel = in_chans // time_span
        self.time_span = time_span
        self.proj = nn.Conv2d(
            self.conv_input_channel,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.apply(self._init_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            C % self.time_span == 0
        ), "Total channels must be divisible by the number of time steps (T)."

        # Reshape input to separate time steps
        x = x.view(B, self.time_span, self.conv_input_channel, H, W)

        # Initialize a list to store projections for each frame
        projected_frames = []

        for t in range(self.time_span):
            # Extract frame t
            frame = x[:, t, :, :, :]

            # Apply projection for the current frame
            proj_frame = self.proj(frame)

            # Flatten spatial dimensions to create patch embeddings
            proj_frame = proj_frame.view(B, -1, proj_frame.size(2) * proj_frame.size(3))
            projected_frames.append(proj_frame)

        # Concatenate all projected frames along the sequence dimension
        x = torch.cat(projected_frames, dim=1)
        return x


class PatchEmbedTimeSeries(nn.Module):

    def __init__(self, patch_size, in_chans, embed_dim, time_span=3):
        super().__init__()
        self.conv_input_channel = in_chans // time_span
        self.time_span = time_span
        self.proj = nn.Conv2d(
            self.conv_input_channel,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.apply(self._init_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            C % self.time_span == 0
        ), "Total channels must be divisible by the number of time steps (T)."

        x = x.view(B, self.time_span, self.conv_input_channel, H, W)
        x = x.view(B * self.time_span, self.conv_input_channel, H, W)
        x = self.proj(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding, from timm"""

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.proj(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("emb", emb)

    def forward(self, x):
        # print(x.shape[1])
        # print(self.emb.shape)
        return self.emb[None, : x.shape[1], :].to(x)


# Predefined Structure
class VisionTransformerTimeSeriesFormalizedWithSAR(nn.Module):
    def __init__(
        self,
        avrg_img_size=320,
        patch_size=10,
        in_chans=1,
        embed_dim=64,
        depth=8,
        num_heads=9,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        global_pool=None,
        gpsa_interval=[-1, -1],
        locality_strength=1.0,
        use_pos_embed=True,
        use_time_embed=False,
        is_LSA=False,
        is_SPT=False,
        is_CSA=False,
        rotary_position_emb=False,
    ):

        super().__init__()
        channel_count = 13
        self.channel_count = channel_count
        self.depth = depth
        self.embed_dim = embed_dim
        embed_dim *= num_heads
        self.num_features = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed
        self.use_time_embed = use_time_embed
        self.is_CSA = is_CSA
        if isinstance(avrg_img_size, int):
            img_size = to_2tuple(avrg_img_size)

        if isinstance(patch_size, int):
            self.patch_size = to_2tuple(patch_size)
        else:
            self.patch_size = patch_size

        self.in_chans = in_chans

        self.PatchDeodceTimeSeries = PatchDecodeTimeSeries(
            patch_size, in_chans, embed_dim, time_span=6
        )
        if not is_SPT:
            if not is_CSA:
                self.patch_embed = PatchEmbed(
                    patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim
                )  # (10,10) -> tuple
            else:
                self.patch_embed = PatchEmbedTimeSeries(
                    patch_size=self.patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    time_span=in_chans // channel_count,
                )

        else:
            self.patch_embed = ShiftedPatchTokenization(
                in_chans, embed_dim, merging_size=self.patch_size, is_pe=True
            )  # (10,10)->tuple

        self.pos_drop = nn.Dropout(p=drop_rate)

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

        if self.use_time_embed:
            self.time_embed = nn.Parameter(
                torch.zeros(
                    1,  # Batch dimension (not used, just a placeholder)
                    embed_dim,  # Embedding dimension
                    self.in_chans // channel_count,  # Temporal height (H_t)
                )
            )

        ### calculating num_patches
        if not is_CSA:
            num_patches = (img_size[0] // self.patch_size[0]) * (
                img_size[1] // self.patch_size[1]
            )
        else:
            num_patches = (
                (img_size[0] // self.patch_size[0])
                * (img_size[1] // self.patch_size[1])
                * (self.in_chans // channel_count)
            )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

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
                    if i >= gpsa_interval[0] - 1 and i < gpsa_interval[1]
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

        self.norm = norm_layer(embed_dim)

        # head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        if not self.is_CSA:
            self.head = nn.Linear(
                self.num_features, in_chans * self.patch_size[0] * self.patch_size[1]
            )
        else:
            self.head = nn.Linear(
                self.num_features,
                channel_count * self.patch_size[0] * self.patch_size[1],
            )

        max_seq_len = 1025
        self.layer_pos_emb = None
        if rotary_position_emb:
            self.layer_pos_emb = FixedPositionalEmbedding(self.embed_dim, max_seq_len)

    def seq2img(self, x, img_size):
        """
        Transforms sequence back into image space, input dims: [batch_size, num_patches, channels]
        output dims: [batch_size, channels, H, W]
        """
        if not self.is_CSA:
            x = x.view(
                x.shape[0],
                x.shape[1],
                self.in_chans,
                self.patch_size[0],
                self.patch_size[1],
            )
            x = x.chunk(x.shape[1], dim=1)
            x = torch.cat(x, dim=4).permute(0, 1, 2, 4, 3)
            x = x.chunk(img_size[0] // self.patch_size[0], dim=3)
            x = torch.cat(x, dim=4).permute(0, 1, 2, 4, 3).squeeze(1)
        else:
            x = x.view(
                x.shape[0],
                x.shape[1],
                self.channel_count,
                self.patch_size[0],
                self.patch_size[1],
            )
            batch_size, num_patches, num_channels, patch_size, _ = x.shape
            patch_per_width = img_size[0] // self.patch_size[0]
            patch_per_height = patch_per_width
            time_span = self.in_chans // self.channel_count
            x = x.view(
                batch_size // time_span,
                time_span,
                num_channels,
                patch_per_width * patch_size,
                patch_per_height * patch_size,
            )
            x = x.view(
                batch_size // time_span,
                num_channels * time_span,
                patch_per_width * patch_size,
                patch_per_height * patch_size,
            )
        return x

        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
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
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B_T, conv_input_channel, H, W)
        if self.use_pos_embed:
            if not self.is_CSA:
                # B: Batch size, E_dim: Embedding dimension, H: Height, W: Width
                _, _, H, W = x.shape  # B, E_dim, H, W
                pos_embed = F.interpolate(
                    self.pos_embed, size=[H, W], mode="bilinear", align_corners=False
                )
                x = x + pos_embed
            else:
                # B_T, conv_input_channel, H, W
                B_T, E_dim, H_p, W_p = x.shape
                T = B_T // B
                pos_embed = self.pos_embed.repeat(
                    B_T, 1, 1, 1
                )  # (B_T, E_dim, H_p, W_p)
                x = x + pos_embed

                # if self.use_time_embed:
                #     # B, E_dim, T]
                #     time_embed = self.time_embed.expand(B, -1, -1)
                #     time_embed_expanded = time_embed.unsqueeze(-1).unsqueeze(
                #         -1
                #     )  # [B, E_dim, T, 1, 1]
                #     time_embed_expanded = time_embed_expanded.expand(
                #         -1, -1, -1, W, W
                #     )  # [B, E_dim, T, H, W]

                #     time_embed_expanded = time_embed_expanded.reshape(B, E_dim, T_H, W)

                #     x = x + time_embed_expanded

        x = x.flatten(2).transpose(1, 2)  # (B_T, seq_len, E_dim)
        x = self.pos_drop(x)
        if self.layer_pos_emb is not None:
            layer_pos_emb = self.layer_pos_emb(x)
        else:
            layer_pos_emb = None

        for u, blk in enumerate(self.blocks):
            x = blk(x, (W, W), pos_emb=layer_pos_emb)

        x = self.norm(x)

        return x

    def forward(self, x):
        _, _, H, W = x.shape  # B, C, H, W
        x = self.forward_features(x)  # B, N_patches, E_dim
        x = self.head(x)  # B, N_patches, Original_channels * Patch_size * Patch_size
        x = self.seq2img(x, (H, W))  #

        return x
