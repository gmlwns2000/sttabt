""" Multi-Scale Vision Transformer v2

@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}

Code adapted from original Apache 2.0 licensed impl at https://github.com/facebookresearch/mvit
Original copyright below.

Modifications and timm support by / Copyright 2022, Ross Wightman
"""
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.
import operator
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial, reduce
from typing import Union, List, Tuple, Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from torch.nn import functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.fx_features import register_notrace_function
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import Mlp, DropPath, trunc_normal_tf_, get_norm_layer, to_2tuple
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'fixed_input_size': True,
        **kwargs
    }


__debug = False
def dlog(*args):
    global __debug
    if __debug:
        print('[DBG]', *args)


default_cfgs = dict(
    mvitv2_tiny=_cfg(url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_T_in1k.pyth'),
    mvitv2_small=_cfg(url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_S_in1k.pyth'),
    mvitv2_base=_cfg(url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in1k.pyth'),
    mvitv2_large=_cfg(url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_L_in1k.pyth'),

    mvitv2_base_in21k=_cfg(
        url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in21k.pyth',
        num_classes=19168),
    mvitv2_large_in21k=_cfg(
        url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_L_in21k.pyth',
        num_classes=19168),
    mvitv2_huge_in21k=_cfg(
        url='https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_H_in21k.pyth',
        num_classes=19168),

    mvitv2_small_cls=_cfg(url=''),
)


@dataclass
class MultiScaleVitCfg:
    depths: Tuple[int, ...] = (2, 3, 16, 3)
    embed_dim: Union[int, Tuple[int, ...]] = 96
    num_heads: Union[int, Tuple[int, ...]] = 1
    mlp_ratio: float = 4.
    pool_first: bool = False
    expand_attn: bool = True
    qkv_bias: bool = True
    use_cls_token: bool = False
    use_abs_pos: bool = False
    residual_pooling: bool = True
    mode: str = 'conv'
    kernel_qkv: Tuple[int, int] = (3, 3)
    stride_q: Optional[Tuple[Tuple[int, int]]] = ((1, 1), (2, 2), (2, 2), (2, 2))
    stride_kv: Optional[Tuple[Tuple[int, int]]] = None
    stride_kv_adaptive: Optional[Tuple[int, int]] = (4, 4)
    patch_kernel: Tuple[int, int] = (7, 7)
    patch_stride: Tuple[int, int] = (4, 4)
    patch_padding: Tuple[int, int] = (3, 3)
    pool_type: str = 'max'
    rel_pos_type: str = 'spatial'
    act_layer: Union[str, Tuple[str, str]] = 'gelu'
    norm_layer: Union[str, Tuple[str, str]] = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        num_stages = len(self.depths)
        if not isinstance(self.embed_dim, (tuple, list)):
            self.embed_dim = tuple(self.embed_dim * 2 ** i for i in range(num_stages))
        assert len(self.embed_dim) == num_stages

        if not isinstance(self.num_heads, (tuple, list)):
            self.num_heads = tuple(self.num_heads * 2 ** i for i in range(num_stages))
        assert len(self.num_heads) == num_stages

        if self.stride_kv_adaptive is not None and self.stride_kv is None:
            _stride_kv = self.stride_kv_adaptive
            pool_kv_stride = []
            for i in range(num_stages):
                if min(self.stride_q[i]) > 1:
                    _stride_kv = [
                        max(_stride_kv[d] // self.stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append(tuple(_stride_kv))
            self.stride_kv = tuple(pool_kv_stride)


model_cfgs = dict(
    mvitv2_tiny=MultiScaleVitCfg(
        depths=(1, 2, 5, 2),
    ),
    mvitv2_small=MultiScaleVitCfg(
        depths=(1, 2, 11, 2),
    ),
    mvitv2_base=MultiScaleVitCfg(
        depths=(2, 3, 16, 3),
    ),
    mvitv2_large=MultiScaleVitCfg(
        depths=(2, 6, 36, 4),
        embed_dim=144,
        num_heads=2,
        expand_attn=False,
    ),

    mvitv2_base_in21k=MultiScaleVitCfg(
        depths=(2, 3, 16, 3),
    ),
    mvitv2_large_in21k=MultiScaleVitCfg(
        depths=(2, 6, 36, 4),
        embed_dim=144,
        num_heads=2,
        expand_attn=False,
    ),

    mvitv2_small_cls=MultiScaleVitCfg(
        depths=(1, 2, 11, 2),
        use_cls_token=True,
    ),
)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
            self,
            dim_in=3,
            dim_out=768,
            kernel=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x) -> Tuple[torch.Tensor, List[int]]:
        x = self.proj(x)
        # B C H W -> B HW C
        return x.flatten(2).transpose(1, 2), x.shape[-2:]


@register_notrace_function
def reshape_pre_pool(
        x,
        feat_size: List[int],
        has_cls_token: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    H, W = feat_size
    if has_cls_token:
        cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]
    else:
        cls_tok = None
    x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
    return x, cls_tok


@register_notrace_function
def reshape_post_pool(
        x,
        num_heads: int,
        cls_tok: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, List[int]]:
    feat_size = [x.shape[2], x.shape[3]]
    L_pooled = x.shape[2] * x.shape[3]
    x = x.reshape(-1, num_heads, x.shape[1], L_pooled).transpose(2, 3)
    if cls_tok is not None:
        x = torch.cat((cls_tok, x), dim=2)
    return x, feat_size


@register_notrace_function
def cal_rel_pos_type(
        attn: torch.Tensor,
        q: torch.Tensor,
        has_cls_token: bool,
        q_size: List[int],
        k_size: List[int],
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_token else 0
    q_h, q_w = q_size
    k_h, k_w = k_size

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class MultiScaleAttentionPoolFirst(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            feat_size,
            num_heads=8,
            qkv_bias=True,
            mode="conv",
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            has_cls_token=True,
            rel_pos_type='spatial',
            residual_pooling=True,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            if kernel_q:
                dlog('pool_q init', kernel_q, stride_q, padding_q)
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = dim // num_heads if mode == "conv" else dim
            if kernel_q:
                self.pool_q = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_v = norm_layer(dim_conv)
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == 'spatial':
            assert feat_size[0] == feat_size[1]
            size = feat_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, feat_size: List[int]):
        B, N, _ = x.shape

        fold_dim = 1 if self.unshared else self.num_heads
        x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
        q = k = v = x

        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)

        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)

        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, v_size = reshape_post_pool(v, self.num_heads, v_tok)
        else:
            v_size = feat_size
        if self.norm_v is not None:
            v = self.norm_v(v)

        q_N = q_size[0] * q_size[1] + int(self.has_cls_token)
        q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
        q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

        k_N = k_size[0] * k_size[1] + int(self.has_cls_token)
        k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
        k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

        v_N = v_size[0] * v_size[1] + int(self.has_cls_token)
        v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
        v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_type == 'spatial':
            attn = cal_rel_pos_type(
                attn,
                q,
                self.has_cls_token,
                q_size,
                k_size,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        self.last_attention_score = attn
        attn = attn.softmax(dim=-1)
        self.last_attention_prob = attn
        raise "not supported"
        dlog('MVIT: last score', self.last_attention_score.shape, 'last prob', self.last_attention_prob.shape, 'pool first')
        x = attn @ v

        if self.residual_pooling:
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x, q_size


class MultiScaleAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            feat_size,
            num_heads=8,
            qkv_bias=True,
            mode="conv",
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            has_cls_token=True,
            rel_pos_type='spatial',
            residual_pooling=True,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        dlog("MPHA:", {
            'dim':dim,
            'dimout':dim_out,
            'kernel_q':kernel_q,
            'kernel_kv': kernel_kv,
            'strideq': stride_q,
            'stridekv':stride_kv,
            'clstk':has_cls_token,
        })

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            if kernel_q:
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            if kernel_q:
                self.pool_q = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    stride_q, #HOTFIX: sttabt
                    stride=stride_q,
                    padding=(0,0), #HOTFIX: sttabt
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_v = norm_layer(dim_conv)
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == 'spatial':
            assert feat_size[0] == feat_size[1]
            size = feat_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

        self.input_mask = None

    def forward(self, x, feat_size: List[int]):
        B, N, _ = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
            dlog('pooled q', self.pool_q)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)

        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
            dlog('pooled k', self.pool_k)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)

        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, _ = reshape_post_pool(v, self.num_heads, v_tok)
            dlog('pooled v', self.pool_v)
        if self.norm_v is not None:
            v = self.norm_v(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_type == 'spatial':
            attn = cal_rel_pos_type(
                attn,
                q,
                self.has_cls_token,
                q_size,
                k_size,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        self.last_attention_score = attn
        if self.input_mask is not None:
            ATTN, ATTH, ATTOUT, ATTIN = attn.shape
            attn_mask = self.input_mask.view(ATTN, 1, 1, ATTIN) * (-10000)
            attn = attn + attn_mask
            # self.input_mask = None
        attn = attn.softmax(dim=-1)
        self.last_attention_prob = attn
        dlog('MVIT: last score ', self.last_attention_score.shape, 'last prob', self.last_attention_prob.shape)
        x = attn @ v

        if self.residual_pooling:
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x, q_size


class MultiScaleBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            num_heads,
            feat_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            mode="conv",
            has_cls_token=True,
            expand_attn=False,
            pool_first=False,
            rel_pos_type='spatial',
            residual_pooling=True,
    ):
        super().__init__()
        proj_needed = dim != dim_out
        self.dim = dim
        self.dim_out = dim_out
        self.has_cls_token = has_cls_token

        self.norm1 = norm_layer(dim)

        self.shortcut_proj_attn = nn.Linear(dim, dim_out) if proj_needed and expand_attn else None
        if stride_q and prod(stride_q) > 1:
            #HOTFIX: reduce local connectivity **for sttabt
            kernel_skip = [s + 1 - 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2)-1 for skip in kernel_skip]
            self.shortcut_pool_attn = nn.MaxPool2d(kernel_skip, stride_skip, padding_skip)
            dlog('Block.shortcut', self.shortcut_pool_attn)
        else:
            self.shortcut_pool_attn = None

        att_dim = dim_out if expand_attn else dim
        # attn_layer = MultiScaleAttentionPoolFirst if pool_first else MultiScaleAttention
        assert not pool_first
        attn_layer = MultiScaleAttention
        self.attn = attn_layer(
            dim,
            att_dim,
            num_heads=num_heads,
            feat_size=feat_size,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_token=has_cls_token,
            mode=mode,
            rel_pos_type=rel_pos_type,
            residual_pooling=residual_pooling,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(att_dim)
        mlp_dim_out = dim_out
        self.shortcut_proj_mlp = nn.Linear(dim, dim_out) if proj_needed and not expand_attn else None
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=int(att_dim * mlp_ratio),
            out_features=mlp_dim_out,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.concrete_init_min = 0.0
        self.concrete_init_max = 0.0
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(self.concrete_init_min, self.concrete_init_max))

        self.input_mask = None
        self.output_mask = None
    
    def set_mask(self, input_mask, output_mask):
        if (input_mask is None) and (output_mask is None):
            self.input_mask = None
            self.output_mask = None
            self.attn.input_mask = None
            self.attn.output_mask = None
            return
        
        # N, TOUT, HID = self.last_output.shape
        # _N, HEAD, _TOUT, TIN = self.attn.last_attention_score.shape
        # assert TOUT == _TOUT
        # assert N == _N
        # assert input_mask.shape == (N, TIN, 1)
        # assert output_mask.shape == (N, TOUT, 1)

        self.input_mask = input_mask
        self.output_mask = output_mask
        self.attn.input_mask = input_mask
        self.attn.output_mask = output_mask

    def _shortcut_pool(self, x, feat_size: List[int]):
        if self.shortcut_pool_attn is None:
            return x
        if self.has_cls_token:
            cls_tok, x = x[:, :1, :], x[:, 1:, :]
        else:
            cls_tok = None
        B, L, C = x.shape
        H, W = feat_size
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.shortcut_pool_attn(x)
        x = x.reshape(B, C, -1).transpose(1, 2)
        if cls_tok is not None:
            x = torch.cat((cls_tok, x), dim=1)

        return x

    def forward(self, x, feat_size: List[int]):
        x_norm = self.norm1(x)
        # NOTE as per the original impl, this seems odd, but shortcut uses un-normalized input if no proj
        x_shortcut = x if self.shortcut_proj_attn is None else self.shortcut_proj_attn(x_norm)
        x_shortcut = self._shortcut_pool(x_shortcut, feat_size)
        # if self.input_mask is not None:
        #     self.input_mask = None
        x, feat_size_new = self.attn(x_norm, feat_size)
        x = x_shortcut + self.drop_path1(x)

        x_norm = self.norm2(x)
        x_shortcut = x if self.shortcut_proj_mlp is None else self.shortcut_proj_mlp(x_norm)
        x = x_shortcut + self.drop_path2(self.mlp(x_norm))

        if self.output_mask is not None:
            x = x * self.output_mask
            # self.output_mask = None

        #backup
        dlog('MVIT: last output', x.shape)
        self.last_output = x

        return x, feat_size_new


class MultiScaleVitStage(nn.Module):

    def __init__(
            self,
            dim,
            dim_out,
            depth,
            num_heads,
            feat_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            mode="conv",
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            has_cls_token=True,
            expand_attn=False,
            pool_first=False,
            rel_pos_type='spatial',
            residual_pooling=True,
            norm_layer=nn.LayerNorm,
            drop_path=0.0,
    ):
        super().__init__()
        self.grad_checkpointing = False

        self.blocks = nn.ModuleList()
        if expand_attn:
            out_dims = (dim_out,) * depth
        else:
            out_dims = (dim,) * (depth - 1) + (dim_out,)

        for i in range(depth):
            attention_block = MultiScaleBlock(
                dim=dim,
                dim_out=out_dims[i],
                num_heads=num_heads,
                feat_size=feat_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                kernel_q=kernel_q,
                kernel_kv=kernel_kv,
                stride_q=stride_q if i == 0 else (1, 1),
                stride_kv=stride_kv,
                mode=mode,
                has_cls_token=has_cls_token,
                pool_first=pool_first,
                rel_pos_type=rel_pos_type,
                residual_pooling=residual_pooling,
                expand_attn=expand_attn,
                norm_layer=norm_layer,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
            )
            dim = out_dims[i]
            self.blocks.append(attention_block)
            if i == 0:
                feat_size = tuple([size // stride for size, stride in zip(feat_size, stride_q)])

        self.feat_size = feat_size

    def forward(self, x, feat_size: List[int]):
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, feat_size = checkpoint.checkpoint(blk, x, feat_size)
            else:
                x, feat_size = blk(x, feat_size)
        return x, feat_size

from models.sparse_token import ApproxSparseBertForSequenceClassificationOutput

class DummyBert:
    def __init__(self, set_concrete_hard_threshold_handler):
        self.set_concrete_hard_threshold_handler = set_concrete_hard_threshold_handler
    
    def set_concrete_hard_threshold(self, x):
        return self.set_concrete_hard_threshold_handler(x)

class MultiScaleVit(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(
            self,
            cfg: MultiScaleVitCfg,
            img_size: Tuple[int, int] = (224, 224),
            in_chans: int = 3,
            global_pool: str = 'avg',
            num_classes: int = 1000,
            drop_path_rate: float = 0.,
            drop_rate: float = 0.,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.config = cfg
        norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.global_pool = global_pool
        self.depths = tuple(cfg.depths)
        self.expand_attn = cfg.expand_attn

        embed_dim = cfg.embed_dim[0]
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.patch_kernel,
            stride=cfg.patch_stride,
            padding=cfg.patch_padding,
        )
        patch_dims = (img_size[0] // cfg.patch_stride[0], img_size[1] // cfg.patch_stride[1])
        num_patches = prod(patch_dims)

        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_prefix_tokens = 1
            pos_embed_dim = num_patches + 1
        else:
            self.num_prefix_tokens = 0
            self.cls_token = None
            pos_embed_dim = num_patches

        if cfg.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))
        else:
            self.pos_embed = None

        num_stages = len(cfg.embed_dim)
        feat_size = patch_dims
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            if cfg.expand_attn:
                dim_out = cfg.embed_dim[i]
            else:
                dim_out = cfg.embed_dim[min(i + 1, num_stages - 1)]
            stage = MultiScaleVitStage(
                dim=embed_dim,
                dim_out=dim_out,
                depth=cfg.depths[i],
                num_heads=cfg.num_heads[i],
                feat_size=feat_size,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                mode=cfg.mode,
                pool_first=cfg.pool_first,
                expand_attn=cfg.expand_attn,
                kernel_q=cfg.kernel_qkv,
                kernel_kv=cfg.kernel_qkv,
                stride_q=cfg.stride_q[i],
                stride_kv=cfg.stride_kv[i],
                has_cls_token=cfg.use_cls_token,
                rel_pos_type=cfg.rel_pos_type,
                residual_pooling=cfg.residual_pooling,
                norm_layer=norm_layer,
                drop_path=dpr[i],
            )
            embed_dim = dim_out
            feat_size = stage.feat_size
            self.stages.append(stage)

        self.num_features = embed_dim
        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(OrderedDict([
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        ]))

        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_tf_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.loss_mode = 'cls'

        #for approx training
        self.main_model = None
        self.factor = 4
        
        #for concrete training
        self.approx_net = None
        self.concrete_hard_threshold = 0.5
        self.concrete_loss_lambda_p = 1e-2
        self.concrete_loss_lambda_mask = 100
        self.loss_fct = None
        
        #for compatibility
        self.bert = DummyBert(
            self.set_concrete_hard_threshold
        )
    
    def reset_concrete_mask(self):
        for stage in self.stages:
            for block in stage.blocks:
                block = block # type: MultiScaleBlock
                block.set_mask(None, None)

    def set_concrete_hard_threshold(self, v):
        self.concrete_hard_threshold = v
    
    def set_concrete_init_p_logit(self, v):
        for stage in self.stages:
            for block in stage.blocks:
                block = block #type: MultiScaleBlock
                block.concrete_init_max = v
                block.concrete_init_min = v
                torch.nn.init.uniform_(block.p_logit, block.concrete_init_min, block.concrete_init_max)
    
    def init_approx_train(self):
        self.transfer_hidden = nn.ModuleList()
        #TODO: support other definition
        hids = [96, 192, 192, 384, 384, 384, 384, 384, 768, 768,]
        for h in hids:
            self.transfer_hidden.append(nn.Linear(h//self.factor, h))

        self.transfer_embedding = nn.Linear(hids[0]//self.factor, hids[0])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_tf_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {k for k, _ in self.named_parameters()
                if any(n in k for n in ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"])}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=[(r'^stages\.(\d+)', None), (r'^norm', (99999,))]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Sequential(OrderedDict([
            ('drop', nn.Dropout(self.drop_rate)),
            ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        ]))

    def forward_features(self, x):
        x, feat_size = self.patch_embed(x)
        B, N, C = x.shape

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        self.last_embedding = x
        dlog('last embedding', self.last_embedding.shape)

        for stage in self.stages:
            x, feat_size = stage(x, feat_size)

        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            if self.global_pool == 'avg':
                x = x[:, self.num_prefix_tokens:].mean(1)
            elif self.global_pool == 'avg1':
                x = x[:, self.num_prefix_tokens:self.num_prefix_tokens+1].mean(1)
            elif self.global_pool == 'avg2':
                x = x[:, self.num_prefix_tokens:self.num_prefix_tokens+2].mean(1)
            elif self.global_pool == 'avg4':
                x = x[:, self.num_prefix_tokens:self.num_prefix_tokens+4].mean(1)
            elif self.global_pool == 'avg8':
                x = x[:, self.num_prefix_tokens:self.num_prefix_tokens+8].mean(1)
            elif self.global_pool == 'avg16':
                x = x[:, self.num_prefix_tokens:self.num_prefix_tokens+16].mean(1)
            else:
                x = x[:, 0]
        return x if pre_logits else self.head(x)

    def update_concrete_mask(self, pixel_values):
        assert self.approx_net is not None
        
        out_approx = self.approx_net(pixel_values)
        approx_scores = get_attention_scores(self.approx_net)

        masks, masks_hard = calc_mvit_concrete_masks(
            approx_scores,
            get_p_logits(self),
            temperature=0.2,
            concrete_hard_threshold=self.concrete_hard_threshold
        )
        
        if self.concrete_hard_threshold is not None:
            masks = masks_hard

        blocks = [] #type: List[MultiScaleBlock]
        for stage in self.stages:
            for block in stage.blocks:
                blocks.append(block)
        
        for i in range(len(approx_scores)):
            blocks[i].set_mask(masks[i], masks[i+1])
        
        dlog('concrete updated')
    
    def forward(self, pixel_values=None, labels=None):
        assert pixel_values is not None

        if self.loss_mode == 'concrete':
            self.reset_concrete_mask()
            self.update_concrete_mask(pixel_values)
        
        x = pixel_values
        x = self.forward_features(x)
        x = self.forward_head(x)
        logits = x
        
        loss_conc_ratio = loss_conc_reg = loss_ltp = loss = 0.0
        loss_att, loss_hid, loss_emb, loss_pred = 0, 0, 0, 0
        if labels is not None:
            if self.loss_mode == 'cls':
                if self.loss_fct is not None:
                    loss = self.loss_fct(x, labels)
                else:
                    loss = F.cross_entropy(x, labels)
            elif self.loss_mode == 'approx':
                main_model = self.main_model #type: MultiScaleVit
                with torch.no_grad():
                    out = main_model(
                        pixel_values=pixel_values,
                        labels=None
                    )
                
                attn = get_attention_probs(self)
                attn_main = get_attention_probs(main_model)

                hid = get_hidden_outputs(self)
                hid_main = get_hidden_outputs(main_model)
                # for i, h in enumerate(hid):
                #     print(h.shape, hid_main[i].shape)

                # from tinybert paper
                # loss attention
                loss_att = 0
                NLAYER = len(attn)
                #from miniLM paper
                for j in range(NLAYER):
                    N, H, TOUT, TIN = attn[j].shape
                    y_pred = attn[j].view(N*H*TOUT, TIN)
                    y_target = attn_main[j].view(N*H*TOUT, TIN)
                    # kl_loss = y_target * ((y_target + EPS).log() - (y_pred + EPS).log())
                    # kl_loss = torch.sum(kl_loss.view(N, H, TOUT, TIN), dim=-1) # shape: N, H, T
                    # kl_loss = torch.mean(kl_loss, dim=-1)
                    # kl_loss = kl_loss.mean() # head and batch mean
                    kl_loss = F.kl_div((y_pred+EPS).log(), y_target, reduction='batchmean')
                    loss_att += kl_loss
                loss_att /= NLAYER
                #loss_att *= 1/100
                
                # loss hidden
                loss_hid = 0
                for j in range(NLAYER):
                    loss_hid += F.mse_loss(
                        self.transfer_hidden[j](hid[j]),
                        hid_main[j]
                    )
                loss_hid /= NLAYER
                loss_hid *= (1/100)
                
                # loss emb
                loss_emb = F.mse_loss(
                    self.transfer_embedding(self.last_embedding), 
                    main_model.last_embedding
                )
                
                # loss prediction
                #print(approx_output.logits[0])
                # loss_pred = F.kl_div(
                #     F.softmax(approx_output.logits, dim=-1),
                #     F.softmax(original_output.logits, dim=-1),
                #     reduction='batchmean'
                # )
                # HOTFIX: LVVIT
                loss_pred = F.mse_loss(
                    F.softmax(logits, dim=-1),
                    F.softmax(out.logits, dim=-1),
                )
                #print(approx_output.logits[0], original_output.logits[0])
                # loss_pred *= 1/10
                # # HOTFIX: LVVIT
                # loss_pred *= 500
                loss_pred *= 200
                # print(loss_pred.item())

                loss = loss_att + loss_hid + loss_emb + loss_pred
            elif self.loss_mode == 'concrete':
                if self.loss_fct is not None:
                    loss = self.loss_fct(x, labels)
                else:
                    loss = F.cross_entropy(x, labels)

                approx_net = self.approx_net #type: MultiScaleVit
                assert approx_net is not None

                blocks = [] #type: List[MultiScaleBlock]
                for stage in self.stages:
                    for block in stage.blocks:
                        blocks.append(block)

                for block in blocks:
                    t = ((block.p_logit - block.concrete_init_min) ** 2) * self.concrete_loss_lambda_p
                    loss_conc_reg += t
                
                target = torch.sigmoid(torch.tensor(
                    blocks[0].concrete_init_min, 
                    device = blocks[0].p_logit.device, 
                    dtype=torch.float32
                ))
                occupy = 0
                occupy += blocks[0].input_mask.mean()
                for block in blocks:
                    occupy += block.output_mask.mean()
                occupy = occupy / (len(blocks) + 1)
                loss_conc_ratio = F.mse_loss(target, occupy) * self.concrete_loss_lambda_mask

                loss = loss + loss_conc_reg + loss_conc_ratio
            else: 
                raise NotImplementedError()
                
        ret = ApproxSparseBertForSequenceClassificationOutput(
            loss=loss,
            logits=x,
            hidden_states=get_hidden_outputs(self),
            attentions=get_attention_probs(self),
            loss_details = {
                'loss_total': loss,
                'loss_ltp': loss_ltp,
                'loss_conc_reg': loss_conc_reg,
                'loss_conc_ratio': loss_conc_ratio,
                'loss_att': loss_att, 
                'loss_hid': loss_hid, 
                'loss_emb': loss_emb, 
                'loss_pred': loss_pred,
            }
        )
        return ret

def get_attention_scores(mvit: MultiScaleVit):
    scores = []
    for stage in mvit.stages:
        stage = stage # type: MultiScaleVitStage
        for block in stage.blocks:
            block = block # type: MultiScaleBlock
            scores.append(block.attn.last_attention_score)
    return scores

def get_attention_probs(mvit: MultiScaleVit):
    scores = []
    for stage in mvit.stages:
        stage = stage # type: MultiScaleVitStage
        for block in stage.blocks:
            block = block # type: MultiScaleBlock
            scores.append(block.attn.last_attention_prob)
    return scores

def get_hidden_outputs(mvit: MultiScaleVit):
    hid = []
    for stage in mvit.stages:
        stage = stage # type: MultiScaleVitStage
        for block in stage.blocks:
            block = block # type: MultiScaleBlock
            hid.append(block.last_output)
    return hid

def get_p_logits(mvit: MultiScaleVit):
    ps = []
    for stage in mvit.stages:
        stage = stage # type: MultiScaleVitStage
        for block in stage.blocks:
            block = block # type: MultiScaleBlock
            ps.append(block.p_logit)
    return ps

def checkpoint_filter_fn(state_dict, model):
    if 'stages.0.blocks.0.norm1.weight' in state_dict:
        return state_dict

    import re
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']

    depths = getattr(model, 'depths', None)
    expand_attn = getattr(model, 'expand_attn', True)
    assert depths is not None, 'model requires depth attribute to remap checkpoints'
    depth_map = {}
    block_idx = 0
    for stage_idx, d in enumerate(depths):
        depth_map.update({i: (stage_idx, i - block_idx) for i in range(block_idx, block_idx + d)})
        block_idx += d

    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(
            r'blocks\.(\d+)',
            lambda x: f'stages.{depth_map[int(x.group(1))][0]}.blocks.{depth_map[int(x.group(1))][1]}',
            k)

        if expand_attn:
            k = re.sub(r'stages\.(\d+).blocks\.(\d+).proj', f'stages.\\1.blocks.\\2.shortcut_proj_attn', k)
        else:
            k = re.sub(r'stages\.(\d+).blocks\.(\d+).proj', f'stages.\\1.blocks.\\2.shortcut_proj_mlp', k)
        if 'head' in k:
            k = k.replace('head.projection', 'head.fc')
        out_dict[k] = v

    return out_dict

def _create_mvitv2(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        MultiScaleVit, variant, pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **kwargs)

# @register_model
def mvitv2_tiny(pretrained=False, **kwargs):
    return _create_mvitv2('mvitv2_tiny', pretrained=pretrained, **kwargs)

def mvitv2_tiny_sttabt(pretrained=False):
    mvit = MultiScaleVit(
        cfg=MultiScaleVitCfg(
            depths=(1, 2, 5, 2),
            kernel_qkv=(1, 1),
            stride_kv_adaptive=(1, 1),
            use_cls_token=True,
        )
    )
    mvit.global_pool = 'nothing'

    if pretrained:
        state = torch.load('./saves/mvit-tiny-deit/checkpoint.pth', map_location='cpu')
        print('mvitv2_tiny_sttabt load from ./saves/mvit-tiny-deit/checkpoint.pth')
        try:
            mvit.load_state_dict(state['model'])
        except RuntimeError as ex:
            print('error during import pretrained weights', ex)
        del state

    return mvit

# Evalute MViT without global pooling
def evalute_model(model, max_steps=987654321):
    import tqdm
    from trainer.vit_approx_trainer import VitApproxTrainer
    trainer = VitApproxTrainer(model='deit-small')

    acc = 0.0
    c = 0
    for i, batch in enumerate(tqdm.tqdm(trainer.timm_data_test)):
        if i > max_steps: break

        batch = {'pixel_values': batch[0].to(trainer.device), 'labels': batch[1].to(trainer.device)}
        inp = batch['pixel_values']
        label = batch['labels']
        
        def accuracy(logits, labels):
            return ((torch.argmax(logits, dim=-1) == labels)*1.0).mean().item()
        
        with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
            acc += accuracy(model(inp), label)
        c += 1
    
    return acc/c

def init_approx_net_from(mvit: MultiScaleVit, factor: int = 4):
    import copy
    cfg = copy.deepcopy(mvit.config)
    cfg.embed_dim = [int(i / factor) for i in cfg.embed_dim]
    print('MVIT: new embd dim', cfg.embed_dim, mvit.config.embed_dim)
    approx_net = MultiScaleVit(
        cfg=cfg,
        global_pool=mvit.global_pool,
    )
    approx_net.main_model = mvit
    approx_net.loss_mode = 'approx'
    approx_net.init_approx_train()
    return approx_net

"""
if __name__ == '__main__':
    img = torch.randn((1, 3, 224, 224))
    mvit = mvitv2_tiny(pretrained=True) #type: MultiScaleVit
    out = mvit(img)
    print(out.shape, mvit.global_pool)

    print('Evalute MViT over limited global average pooling...')

    mvit = mvit.to(0)
    mvit.global_pool = 'avg'
    print('mvitv2 acc all:', evalute_model(mvit))
    
    mvit.global_pool = 'nothing'
    print('mvitv2 acc nothing:', evalute_model(mvit))

    mvit.global_pool = 'avg1'
    print('mvitv2 acc avg1:', evalute_model(mvit))

    mvit.global_pool = 'avg2'
    print('mvitv2 acc avg2:', evalute_model(mvit))

    mvit.global_pool = 'avg4'
    print('mvitv2 acc avg4:', evalute_model(mvit))

    mvit.global_pool = 'avg8'
    print('mvitv2 acc avg8:', evalute_model(mvit))

    mvit.global_pool = 'avg16'
    print('mvitv2 acc avg16:', evalute_model(mvit))
"""

import models.sparse_token as sparse
from models.sparse_token import raise_if_nan, STANDARD_NORMAL_DISTRIBUTION

EPS = 1e-7

import copy
from typing import List

def calc_mvit_concrete_masks(
    attention_scores: "List[torch.Tensor]",
    p_logits: "List[torch.Tensor]",
    temperature=0.1,
    concrete_hard_threshold=0.5,
):
    __abt_default_p = sparse.__abt_default_p
        
    #update mask
    output_masks = []
    output_masks_hard = []

    TLEN = attention_scores[-1].shape[-2]
    N = attention_scores[-1].shape[0]
    device = attention_scores[0].device
    last_score = torch.zeros((1, TLEN), dtype=torch.float32, device=device)
    last_concrete_score = torch.zeros((1, TLEN), dtype=torch.float32, device=device)
    last_mask = torch.zeros((1, TLEN), dtype=torch.float32, device=device)
    last_score[0, 0] = 1.0
    last_concrete_score[0, 0] = 1.0
    last_mask[0, 0] = 1.0
    last_mask = last_mask.expand((N, -1))
    last_mask_un = last_mask.view(-1, TLEN, 1)

    output_masks.append(last_mask_un)
    output_masks_hard.append(last_mask_un)

    if sparse.BENCHMARK_CONCRETE_OCCUPY: 
        with torch.no_grad():
            sparse.benchmark_cum('concrete_occupy', last_mask_un.mean())
    
    for j in range(len(attention_scores)):
        #layer indexing
        i = len(attention_scores) - j - 1
        attention_score = attention_scores[i]
        p_logit = p_logits[i]
        
        att_score = attention_score #N, H, TOUT, TIN
        N, H, TOUT, TIN = att_score.shape
        raise_if_nan(att_score)
        att_score_masked = att_score
        raise_if_nan(att_score_masked)
        att_score_mean = torch.mean(att_score_masked, dim=-1, keepdim=True)
        raise_if_nan(att_score_mean)
        att_score_var = torch.mean(torch.square((att_score_masked - att_score_mean)), dim=-1, keepdim=True)
        raise_if_nan(att_score_var)
        att_score_std = torch.sqrt(att_score_var)
        raise_if_nan(att_score_std)
        std_att_score = (att_score - att_score_mean) / (att_score_std + EPS)
        raise_if_nan(std_att_score)
        uni_att_score = STANDARD_NORMAL_DISTRIBUTION.cdf(std_att_score) #torch.distributions.Normal(0, 1).cdf(std_att_score)
        uni_att_score = torch.mean(uni_att_score, dim=1) # head

        N, T, _ = uni_att_score.shape
        uni_att_score = uni_att_score * last_mask.unsqueeze(-1)
        score_prop = __abt_default_p
        uni_att_score = torch.sum(
            uni_att_score * last_concrete_score.unsqueeze(-1) * score_prop + uni_att_score * (1-score_prop), dim=1
        ) / (torch.sum(last_mask, dim=1, keepdim=True) + EPS)
        raise_if_nan(uni_att_score)
        
        uni_att_score = uni_att_score / (torch.max(uni_att_score, dim=-1, keepdim=True)[0] + EPS)
        raise_if_nan(uni_att_score)
        empty_base = 0.01
        uni_att_score = (empty_base + (1-empty_base) * uni_att_score)
        concrete_score = uni_att_score
        last_concrete_score = concrete_score
        
        p = torch.sigmoid(p_logit).view(1, 1)
        mask = torch.sigmoid((torch.log(p + EPS) - torch.log(1 - p + EPS) + torch.log(concrete_score + EPS) - torch.log(1 - concrete_score + EPS)) / (temperature))
        raise_if_nan(mask)

        if mask.shape[1] > last_mask.shape[1]:
            NN, NT = mask.shape
            LN, LT = last_mask.shape
            assert LN == NN

            tokens_len = LT-1
            isize = int(tokens_len**0.5)
            assert isize**2 == tokens_len

            new_isize = isize*2
            assert NT == (new_isize**2 + 1)

            timg = torch.zeros((LN, new_isize, new_isize), dtype=last_mask.dtype, device=last_mask.device)
            #consider conv2x2 and maxpool2x2
            timg[:, ::2, ::2] = last_mask[:,1:].view(LN, isize, isize)
            timg[:, 0::2, 1::2] = last_mask[:,1:].view(LN, isize, isize)
            timg[:, 1::2, 0::2] = last_mask[:,1:].view(LN, isize, isize)
            timg[:, 1::2, 1::2] = last_mask[:,1:].view(LN, isize, isize)

            # dlog('help_pre', last_mask[:,1:].view(LN, isize, isize)[0])
            # dlog('help after', timg[0])

            tmask = torch.zeros((NN, NT), dtype=mask.dtype, device=mask.device)
            tmask[:,0] = 1.0
            tmask[:,1:] = timg.view(LN, -1)
            last_mask = tmask

            assert mask.shape == last_mask.shape
        elif mask.shape[1] == last_mask.shape[1]:
            pass
        else:
            raise "token must be increasing while back-tracking"
        
        current_mask = torch.max(torch.stack([mask, last_mask], dim=0), dim=0)[0]

        assert current_mask.shape == (N, TIN)
        
        current_mask_un = current_mask.unsqueeze(-1)
        current_mask_hard = None
        if concrete_hard_threshold is not None:
            current_mask_hard = (current_mask_un >= concrete_hard_threshold) * 1.0
            if sparse.BENCHMARK_CONCRETE_OCCUPY:
                with torch.no_grad(): sparse.benchmark_cum('concrete_occupy', current_mask_hard.mean())
        else:
            if sparse.BENCHMARK_CONCRETE_OCCUPY:
                with torch.no_grad(): sparse.benchmark_cum('concrete_occupy', current_mask_un.mean())
        
        output_masks.append(current_mask_un)
        output_masks_hard.append(current_mask_hard)

        last_mask = current_mask
    
    output_masks.reverse()
    output_masks_hard.reverse()
    
    return output_masks, output_masks_hard

def mvitv2_tiny_sttabt_concrete(pretrained=False, approx_net=None, factor=4):
    mvit = mvitv2_tiny_sttabt(pretrained=pretrained)
    if approx_net is None:
        approx_net = init_approx_net_from(copy.deepcopy(mvit), factor=factor)
        assert factor == 4
        
        # state = torch.load('./saves/vit-approx-mvit-tiny-in1k-f4.pth', map_location='cpu')
        # try:
        #     if list(state['model'].keys())[0].startswith('module.'):
        #         from utils import ddp
        #         wrapper = ddp.MimicDDP(approx_net)
        #         wrapper.load_state_dict(state['model'])
        #     else:
        #         approx_net.load_state_dict(state['model'])
        # except RuntimeError as ex:
        #     print('error while load approxnet', ex)
        # del state
        state = torch.load('./saves/mvit-tiny-deit-approx/checkpoint.pth', map_location='cpu')
        print('mvitv2_tiny_sttabt_concrete load from ./saves/mvit-tiny-deit-approx/checkpoint.pth')
        approx_net.load_state_dict(state['model'])
        del state
    
    mvit.approx_net = approx_net
    mvit.loss_mode = 'concrete'

    return mvit

if __name__ == '__main__':
    __debug = True

    img = torch.randn((1, 3, 224, 224))
    mvit = mvitv2_tiny_sttabt() #type: MultiScaleVit
    approx_net = init_approx_net_from(mvit)

    out = mvit(img)
    out_approx = approx_net(img)

    assert out.logits.shape == out_approx.logits.shape
    print(out.logits.shape)

    print('load pretrained')
    mvitv2_tiny_sttabt(pretrained=True)

    attention_scores = []
    p_logits = []
    for stage in mvit.stages:
        stage = stage #type: MultiScaleVitStage
        for block in stage.blocks:
            block = block #type: MultiScaleBlock
            attention_scores.append(block.attn.last_attention_score)
            p_logits.append(block.p_logit)
            dlog(f'attention_scores[{len(attention_scores)-1}]', attention_scores[-1].shape)
    
    print('layers:', len(attention_scores))

    #test mask shapes
    output_mask, output_hard_mask = calc_mvit_concrete_masks(
        attention_scores,
        p_logits,
        concrete_hard_threshold=None,
    )
    for i, mask in enumerate(output_mask):
        print(f'concrete mask[{i}]', mask.shape)

    #test mask for different p
    for p in [-2.5, -1.5, 0.0, 1.0, 1.5]:
        sparse.benchmark_reset()
        mvit.set_concrete_init_p_logit(p)
        #test soft mask
        output_mask, output_hard_mask = calc_mvit_concrete_masks(
            attention_scores,
            p_logits,
            concrete_hard_threshold=None,
        )
        print('concrete_occupy', p, sparse.benchmark_get_average('concrete_occupy'), len(output_mask))

        #test hard mask
        output_mask, output_hard_mask = calc_mvit_concrete_masks(
            attention_scores,
            p_logits,
            concrete_hard_threshold=0.5,
        )
        print('concrete_occupy', p, sparse.benchmark_get_average('concrete_occupy'))
    
    #test concrete masking
    mvit.approx_net = approx_net
    mvit.loss_mode = 'concrete'

    mvit.set_concrete_init_p_logit(-2)
    mvit.set_concrete_hard_threshold(None)
    sparse.benchmark_reset()
    mvit(img)
    print('concrete_occupy', p, sparse.benchmark_get_average('concrete_occupy'))

    mvit.set_concrete_hard_threshold(0.5)
    sparse.benchmark_reset()
    mvit(img)
    print('concrete_occupy', p, sparse.benchmark_get_average('concrete_occupy'))

    #test concrete loss
    mvit = mvitv2_tiny_sttabt_concrete(approx_net=None, factor=4)
    mvit.set_concrete_hard_threshold(0.5)
    sparse.benchmark_reset()
    out = mvit(img, torch.ones((img.shape[0]), dtype=torch.long))
    print('concrete_loss', out['loss'], out['loss_details'])

    #example optimize
    img = torch.randn((16, 3, 224, 224))
    labels = torch.arange(0, img.shape[0])

    mvit = mvitv2_tiny_sttabt_concrete(approx_net=None, factor=4)
    mvit.set_concrete_hard_threshold(None)

    mvit = mvit.to(0)
    mvit.train()
    img = img.to(0)
    labels = labels.to(0)

    optimizer = torch.optim.Adam(mvit.parameters(), lr=1e-1)
    
    __debug = False
    import tqdm
    for i in tqdm.trange(1000):
        sparse.benchmark_reset()
        out = mvit(img, labels)
        
        optimizer.zero_grad()
        out['loss'].backward()
        optimizer.step()

        if (i%10) == 0:
            print(i, out['loss'], out['loss_details'], sparse.benchmark_get_average('concrete_occupy'))
    
    __debug = True