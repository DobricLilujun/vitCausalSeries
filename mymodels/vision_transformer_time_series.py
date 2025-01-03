"""
Code borrowed from https://github.com/facebookresearch/convit which uses code from timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

Modifications include adaptation to image reconstruction, variable input sizes, and patch sizes for both dimensions.
"""

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils.spt import ShiftedPatchTokenization
from torch import einsum
from einops import rearrange, repeat


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


class MHSA_time_series(nn.Module):
    def __init__(
        self, dim, num_patches, heads=8, dim_head=64, dropout=0.0, is_LSA=False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_time_steps = 3
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        init_weights(self.to_qkv)
        self.to_out = (
            nn.Sequential(nn.Linear(self.inner_dim, self.dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        is_time_series = True
        # pos_emb = None
        #  mask = 1 允许，mask = 0 屏蔽
        if is_LSA:  ## Local Self Attention
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.eye(
                self.num_patches, self.num_patches
            )  # auxilary one for cls
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        elif is_time_series:
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.zeros(
                self.num_patches * self.num_time_steps,
                self.num_patches * self.num_time_steps,
            )
            # Spatial Attention
            for t in range(self.num_time_steps):
                cur_time_start_idx = t * self.num_patches
                cur_time_end_idx = cur_time_start_idx + self.num_patches
                self.mask[
                    cur_time_start_idx:cur_time_end_idx,
                    cur_time_start_idx:cur_time_end_idx,
                ] = 1

            # 填充mask：时间步之间的空间注意力（相邻的时间步之间有注意力）
            for t in range(1, self.num_time_steps):
                start_idx = t * self.num_patches
                prev_idx = (t - 1) * self.num_patches
                self.mask[
                    start_idx : start_idx + self.num_patches,
                    prev_idx : prev_idx + self.num_patches,
                ] = 1  # 向前相邻的时间步

                next_idx = (t + 1) * self.num_patches
                if t < self.num_time_steps - 1:
                    self.mask[
                        start_idx : start_idx + self.num_patches,
                        next_idx : next_idx + self.num_patches,
                    ] = 1  # 向后相邻的时间步

                    # self.mask[prev_time_start_idx:prev_time_end_idx, cur_time_start_idx:cur_time_end_idx] = 1

            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x, pos_emb=None):
        # (batch size, flat_patch_dim , embed_dim，Head_number)
        b, n, _, h = *x.shape, self.heads

        # q,k,v (batch size b, flat_patch_dim n,inner_dim(heads h * dim_head d))
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        pos_emb = None
        if exists(pos_emb):
            q, k = apply_rotary_pos_emb(q, k, pos_emb)  # Apply the positional embedding

        if self.mask is None:
            dots = (
                einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
            )  # Calculate the dot product # (batch_size b, heads h, flat_patch_dim n , flat_patch_dim n)

        else:
            scale = self.scale
            dots = torch.mul(
                einsum("b h i d, b h j d -> b h i j", q, k),
                scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)),
            )
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # (batch size, heads, height × width , dim_head)

        out = rearrange(
            out, "b h n d -> b n (h d)"
        )  #  (b,  flat_patch_dim n, inner_dim)
        return self.to_out(
            out
        )  #  (b,  flat_patch_dim n, inner_dim) -> (b, n, dim_embedding)


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
        **kwargs
    ):
        super().__init__()
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
            self.attn = MHSA_time_series(
                dim,
                num_patches,
                heads=num_heads,
                dropout=drop,
                is_LSA=False,
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

    def forward(self, x, grid_size, pos_emb=None):
        # self.attn.current_grid_size = grid_size
        # (batch b, flatten_patch_dim n, dim_embedding)
        x = x + self.drop_path(self.attn(self.norm1(x), pos_emb=pos_emb))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # (batch b, flatten_patch_dim n, dim_embedding)
        return x


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


class PatchEmbedTimeSerie(nn.Module):

    def __init__(self, patch_size, in_chans, embed_dim, time_span=3):
        super().__init__()
        self.conv_input_channel = in_chans // time_span
        self.time_span = time_span
        # 先用一个conv
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

        # Flatten spatial dimensions to patches
        B_T, embed_dim, H_p, W_p = x.shape
        x = x.flatten(2)  # Shape: (B*T, embed_dim, N), where N = H_p * W_p
        x = x.view(B, self.time_span, -1, embed_dim)  # Shape: (B, T, N, embed_dim)
        x = x.view(
            B, self.time_span * x.size(2), embed_dim
        )  # Shape: (B, T*N, embed_dim)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# Rotational Position Embedding for Vision Transformers
# 分析和收集多尺度频率信息
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, time_span=3):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.time_span = time_span
        self.register_buffer("emb", emb)

    # B * T_N * D
    def forward(self, x):
        B, T_N, D = x.shape
        assert T_N % self.time_span == 0, "T_N must be divisible by time_span"

        N = T_N // self.time_span
        T = self.time_span
        pos_emb = self.emb[None, :N, :].to(x)
        pos_embs = [pos_emb for i in range(T)]
        pos_emb = torch.cat(pos_embs, dim=1)
        return pos_emb


# Predefined Structure
class VisionTransformerTimeSeries(nn.Module):
    """Vision Transformer"""

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
        is_LSA=False,
        is_SPT=False,
        rotary_position_emb=False,
        time_span=3,
    ):

        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        embed_dim *= num_heads
        self.num_features = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed
        self.time_span = time_span
        if isinstance(avrg_img_size, int):
            img_size = to_2tuple(avrg_img_size)

        if isinstance(patch_size, int):
            self.patch_size = to_2tuple(patch_size)
        else:
            self.patch_size = patch_size

        self.in_chans = in_chans

        if not is_SPT:
            self.patch_embed = PatchEmbedTimeSerie(
                patch_size=self.patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                time_span=time_span,
            )  # (10,10) -> tuple
        else:
            # Need to keep when I have time
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

        ### calculating num_patches
        num_patches = (
            (img_size[0] // self.patch_size[0])
            * (img_size[1] // self.patch_size[1])
            * self.time_span
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
                    )
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module="head")]
        self.head = nn.Linear(
            self.num_features,
            in_chans
            // self.patch_embed.time_span
            * self.patch_size[0]
            * self.patch_size[1],
        )
        max_seq_len = 1025
        self.layer_pos_emb = None
        if rotary_position_emb:
            self.layer_pos_emb = FixedPositionalEmbedding(
                self.embed_dim, max_seq_len, time_span=self.patch_embed.time_span
            )

    def seq2img(self, x, img_size):
        """
        Transforms sequence back into image space, input dims: [batch_size (10), flatten_patch_dim (6*6*3), channels_each (11) * patch_size[0] (10) * patch_size[1] (10)]
        output dims: [batch_size, channels_each, H, W]
        """
        img_height, img_width = img_size
        num_patch_line = img_height // self.patch_size[0]
        time_span = self.patch_embed.time_span
        num_images = time_span
        num_channels = self.in_chans // time_span
        x = x.view(
            x.shape[0],  # 10
            num_patch_line * num_patch_line * time_span,
            self.in_chans // time_span,  # 11
            self.patch_size[0],  # 10
            self.patch_size[1],  # 10
        )  # [batch_size, num_patches, channels_each, patch_size, patch_size]

        batch_size = x.shape[0]
        num_patches = x.shape[1]
        patch_height, patch_width = self.patch_size[0], self.patch_size[1]

        patches_per_image = (img_height // patch_height) * (img_width // patch_width)
        assert (
            num_patches == patches_per_image * self.time_span
        ), "Mismatch between patches and images."

        x = x.view(
            batch_size,
            self.time_span,
            patches_per_image,
            num_channels,
            patch_height,
            patch_width,
        )
        x = x.view(
            batch_size,
            self.time_span,
            int(img_height / patch_height),
            int(img_width / patch_width),
            num_channels,
            patch_height,
            patch_width,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(batch_size, num_images * num_channels, img_height, img_width)
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
            self.num_features,
            self.in_chans
            // self.patch_embed.time_span
            * self.patch_size[0]
            * self.patch_size[1],
        )

    # 把两个pos_embedding 搞清楚 把兼容time series 的弄出来
    def forward_features(self, x):

        # -------------------------------------------------Patch Embedding For Time Series Images  -----------------------------------------
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        B, N, D = x.shape  # bactch * (patch_num_per_image * time_span) * embed_dim
        T = self.time_span
        num_patches_per_T = N // T
        num_patchs_line = int((num_patches_per_T) ** 0.5)
        H_p, W_p = num_patchs_line, num_patchs_line

        # -------------------------------------------------Add Positional Embedding For Time Series Images  -----------------------------------------
        if self.use_pos_embed:
            pos_embed = F.interpolate(
                self.pos_embed,
                size=[H_p, W_p],
                mode="bilinear",
                align_corners=False,
            )
            x_split = x.view(B, T, num_patches_per_T, D)  # (B, T, num_patches_per_T, D)
            x_split = x_split.permute(1, 0, 3, 2).view(
                T, B, D, H_p, W_p
            )  # (T, B, D, H_p, W_p)

            # Add positional embeddings to each time step
            x_with_pos = []
            for t in range(T):
                x_t = x_split[t]
                x_t_with_pos = x_t + pos_embed  # (B, D, H_p, W_p)
                x_with_pos.append(x_t_with_pos)

            x = torch.stack(x_with_pos, dim=0)  # B, D, H_p, W_p) -> (T, B, D, H_p, W_p)
            x = x.permute(1, 0, 2, 3, 4)  # (T, B, D, H_p, W_p) -> (B, T, D, H_p, W_p)
            x = x.reshape(B, N, D)  # (B, T, D, H_p, W_p) -> (B, N (T*H_p*W_p), D)

        ## -------------------------------------------------   Format input images to  (batch b, flatten_patch_dim n, embed_dim) for transformer  -----------------------------------------
        # x = x.flatten(2).transpose(
        #     1, 2
        # )  # (B, N, D) -> (B, D, N)  # (batch b, flatten_patch_dim n, embed_dim)
        x = self.pos_drop(x)  # Random drop

        ## -------------------------------------------------  Rotary Position Embedding For Time Series Images  ------------------------------------------------
        #  (batch size, flatten_patch_dim n ,embed_dim)  相对位置更高频率级别的embedding
        if self.layer_pos_emb is not None:
            layer_pos_emb = self.layer_pos_emb(x)
        else:
            layer_pos_emb = None

        ## -------------------------------------------------  Transformer Block ------------------------------------------------
        # (batch size, flatten_patch_dim n ,embed_dim)
        for u, blk in enumerate(self.blocks):
            x = blk(x, (H, W), pos_emb=layer_pos_emb)

        # (batch b, flatten_patch_dim n, embed_dim)
        x = self.norm(x)

        return x

    def forward(self, x):
        _, _, H, W = x.shape  ## (Batch b, Channel c, Height h, Width w) time_Series 33
        x = self.forward_features(
            x
        )  # Batch Size * (Patch_num_per_image * time_span) * embed_dim

        # nn.Linear(
        #     self.num_features,
        #     in_chans
        #     // self.patch_embed.time_span
        #     * self.patch_size[0]
        #     * self.patch_size[1],
        # )

        x = self.head(
            x
        )  #  (batch, flatten_patch_dim n, embed_dim) -> (batch, flatten_patch_dim n, in_chans * self.patch_size[0] * self.patch_size[1])
        x = self.seq2img(
            x, (H, W)
        )  # (batch, flatten_patch_dim n, in_chans * self.patch_size[0] * self.patch_size[1])  -> [batch_size, channels, H, W]
        return x  # [batch_size, channels, H, W]
