import warnings
import torch.nn.functional as F
from functools import partial

import torchvision.models
from timm.models.layers import to_2tuple, trunc_normal_
import math
from timm.models.layers import DropPath
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn
import torch


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        #        x = self.head(x[3])

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DownConv(nn.Module):
    """ Down sampling Feature Maps"""

    def __init__(self, channel=256):
        super().__init__()

        self.downsample = nn.Sequential(
            # conv 尺寸不变 （256 * 256）
            nn.Conv2d(in_channels=channel, out_channels=channel, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=channel),

            # conv 尺寸减小 8 倍（32 * 32）
            nn.Conv2d(in_channels=channel, out_channels=channel * 2, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=(channel * 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel * 2, out_channels=channel * 2, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=(channel * 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel * 2, out_channels=channel * 2, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=(channel * 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel * 2, out_channels=channel * 2, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=(channel * 2)),

            # conv 尺寸减小 8 倍（4 * 4）
            nn.Conv2d(in_channels=channel * 2, out_channels=channel * 4, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=(channel * 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel * 4, out_channels=channel * 4, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=(channel * 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel * 4, out_channels=channel * 4, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=(channel * 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel * 4, out_channels=channel * 4, padding=1, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=(channel * 4)),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(nn.Linear(1024, 3))

    def forward(self, x):
        x = self.downsample(x)

        x = self.avg(x)
        # x = torch.flatten(x, 1)
        # f = self.fc(x)

        return x


class AttentionWeight(Module):
    def __init__(self, channels_in=1024, channels_out=4, channels_embedding=256):
        super(AttentionWeight, self).__init__()
        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels=channels_in, out_channels=channels_embedding, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=channels_embedding, out_channels=channels_out, kernel_size=3, padding=1),
            nn.Softmax(dim=1))

    def forward(self, inputs):
        outs = self.down_channel(inputs)
        return outs


class ClsPred(Module):
    def __init__(self, in_channel=1024):
        super(ClsPred, self).__init__()
        self.pred = nn.Sequential(nn.Linear(in_channel, 3))

    def forward(self, x):
        f = torch.flatten(x, 1)
        f = self.pred(f)
        return f


class ClsWeight1(Module):
    def __init__(self, channel=256, embed_dim=512):
        super(ClsWeight1, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=embed_dim, kernel_size=1),
                                  nn.Conv2d(in_channels=embed_dim, out_channels=channel, kernel_size=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        f = self.conv(x)
        f = self.softmax(f)
        return f


class ClsWeight2(Module):
    def __init__(self, channel=256, embed_dim=512):
        super(ClsWeight2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=embed_dim, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=embed_dim, out_channels=channel, kernel_size=3, padding=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        f = self.conv(x)
        f = self.softmax(f)
        return f


class ClsWeight3(Module):
    def __init__(self, channel=256, embed_dim=512):
        super(ClsWeight3, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=embed_dim, kernel_size=5, padding=2),
                                  nn.Conv2d(in_channels=embed_dim, out_channels=channel, kernel_size=5, padding=2))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        f = self.conv(x)
        f = self.softmax(f)
        return f


class ClsWeight4(Module):
    def __init__(self, channel=256, embed_dim=512):
        super(ClsWeight4, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=embed_dim, kernel_size=7, padding=3),
                                  nn.Conv2d(in_channels=embed_dim, out_channels=channel, kernel_size=7, padding=3))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        f = self.conv(x)
        f = self.softmax(f)
        return f


class SegWeight(Module):
    def __init__(self, channel=1024):
        super(SegWeight, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
                                  nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        f = self.conv(x)

        f1 = self.softmax(f[:, 0:256, :, :])
        f2 = self.softmax(f[:, 256:512, :, :])
        f3 = self.softmax(f[:, 512:768, :, :])
        f4 = self.softmax(f[:, 768:1024, :, :])

        return f1, f2, f3, f4


class Decoder(Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, dims, dim, class_num=2):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.weight_cls = AttentionWeight()
        self.weight_seg = AttentionWeight()

        self.down_conv = DownConv(channel=256)
        # 分割指导分类的权重
        self.cls_weight1 = ClsWeight1()
        self.cls_weight2 = ClsWeight2()
        self.cls_weight3 = ClsWeight3()
        self.cls_weight4 = ClsWeight4()

        # 分类指导分割的权重
        self.seg_weight = SegWeight()

        self.cls_pred = ClsPred(in_channel=1024)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        # 全局特征的变换，全部都变换到 H * W * C = 256 * 256 * 256
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # 将得到的特征图抽取出来，进行拼接用于获得权重特征图
        feature_map = torch.cat((_c1, _c2, _c3, _c4), dim=1)
        seg_attn = self.weight_seg(feature_map)
        cls_attn = self.weight_cls(feature_map)

        # 用于分类的特征图
        c4_cls = torch.unsqueeze(cls_attn[:, 3, :, :], dim=1).repeat_interleave(256, dim=1) * _c4
        c3_cls = torch.unsqueeze(cls_attn[:, 2, :, :], dim=1).repeat_interleave(256, dim=1) * _c3
        c2_cls = torch.unsqueeze(cls_attn[:, 1, :, :], dim=1).repeat_interleave(256, dim=1) * _c2
        c1_cls = torch.unsqueeze(cls_attn[:, 0, :, :], dim=1).repeat_interleave(256, dim=1) * _c1

        # 用于分割的特征图
        c4_seg = torch.unsqueeze(seg_attn[:, 3, :, :], dim=1).repeat_interleave(256, dim=1) * _c4
        c3_seg = torch.unsqueeze(seg_attn[:, 2, :, :], dim=1).repeat_interleave(256, dim=1) * _c3
        c2_seg = torch.unsqueeze(seg_attn[:, 1, :, :], dim=1).repeat_interleave(256, dim=1) * _c2
        c1_seg = torch.unsqueeze(seg_attn[:, 0, :, :], dim=1).repeat_interleave(256, dim=1) * _c1

        # 分割预测分支，输出 1 x 256 x 256 x 256
        L34 = self.linear_fuse34(torch.cat([c4_seg, c3_seg], dim=1))
        L2 = self.linear_fuse2(torch.cat([L34, c2_seg], dim=1))
        seg_feature = self.linear_fuse1(torch.cat([L2, c1_seg], dim=1))

        # 分类预测分支，输出 1 x 1024 x 1 x 1
        cls_feature = self.down_conv(c1_cls + c2_cls + c3_cls + c4_cls)

        # 分割的 feature map 指导分类
        c1_cls += seg_feature * self.cls_weight1(seg_feature)
        c2_cls += seg_feature * self.cls_weight2(seg_feature)
        c3_cls += seg_feature * self.cls_weight3(seg_feature)
        c4_cls += seg_feature * self.cls_weight4(seg_feature)

        # 分类的 feature map 指导分割
        seg_weight1, seg_weight2, seg_weight3, seg_weight4 = self.seg_weight(cls_feature)
        c1_seg += seg_weight1 * c1_seg
        c2_seg += seg_weight2 * c2_seg
        c3_seg += seg_weight3 * c3_seg
        c4_seg += seg_weight4 * c4_seg

        # 使用更新后的分割和分类 feature map
        L34 = self.linear_fuse34(torch.cat([c4_seg, c3_seg], dim=1))
        L2 = self.linear_fuse2(torch.cat([L34, c2_seg], dim=1))
        seg_feature = self.linear_fuse1(torch.cat([L2, c1_seg], dim=1))

        # 分类预测分支，输出 1 x 1024 x 1 x 1
        cls_feature = self.down_conv(c1_cls + c2_cls + c3_cls + c4_cls)
        cls_outcome = self.cls_pred(cls_feature)

        seg_feature = self.dropout(seg_feature)
        seg_outcome = self.linear_pred(seg_feature)

        return seg_outcome, cls_outcome


class TSNet(nn.Module):
    def __init__(self, class_num=2, **kwargs):
        super(TSNet, self).__init__()
        self.class_num = class_num
        self.backbone = mit_b2()
        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=256, class_num=class_num)
        # self._init_weights()  # load pretrain

    def forward(self, x):
        features = self.backbone(x)

        features, c4_down_conv = self.decode_head(features)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(features)
        return {"out": features, "cls": c4_down_conv}

#     def _init_weights(self):
#         pretrained_dict = torch.load('/mnt/DATA-1/DATA-2/Feilong/scformer/models/mit/mit_b2.pth')
#         model_dict = self.backbone.state_dict()
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dicdt}
#         model_dict.update(pretrained_dict)
#         self.backbone.load_state_dict(model_dict)
#         print("successfully loaded!!!!")


if __name__ == '__main__':
    print('hello, python!')

    model = TSNet(class_num=2)
    x = torch.ones(size=(1, 1, 1024, 1024))
    y = model(x)
    print(y['out'].shape, y['cls'].shape)


