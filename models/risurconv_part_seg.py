"""
Author: Zhiyuan Zhang
Date: July 2024
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from risurconv_utils import RISurConvSetAbstraction
from time import time
from timm.models.layers import DropPath, trunc_normal_
from risurconv_utils import sample_and_group

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 64
        self.category_num=16
        self.normal_channel = normal_channel
        
        self.sa0 = RISurConvSetAbstraction(npoint=512, radius=0.2,  nsample=8, in_channel= 0, out_channel=64,  group_all=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(128, num_class, 1)

        self.npoint=512
        self.radius=0.2
        self.nsample=8

        self.sa0 = RISurConvSetAbstraction(npoint=512, radius=0.2,  nsample=8, in_channel= 0, out_channel=384,  group_all=False)

        #以下是pointmae的部分
        self.mae_trans_dim = 384
        self.mae_depth = 12
        self.mae_drop_path_rate = 0.1
        self.mae_cls_dim = num_class
        self.mae_num_heads = 6

        self.mae_group_size = 32
        self.mae_num_group = 128
        # define the encoder
        self.mae_encoder_dims = 384

        self.mae_pos_embed = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, self.mae_trans_dim)
        )

        mae_dpr = [x.item() for x in torch.linspace(0, self.mae_drop_path_rate, self.mae_depth)]
        self.mae_blocks = TransformerEncoder(
            embed_dim=self.mae_trans_dim,
            depth=self.mae_depth,
            drop_path_rate=mae_dpr,
            num_heads=self.mae_num_heads
        )

        self.mae_norm = nn.LayerNorm(self.mae_trans_dim)

        self.mae_label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.mae_propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.mae_trans_dim * 4, 1024])

        self.mae_convs1 = nn.Conv1d(3392, 512, 1)
        self.mae_dp1 = nn.Dropout(0.5)
        self.mae_convs2 = nn.Conv1d(512, 256, 1)
        self.mae_convs3 = nn.Conv1d(256, self.mae_cls_dim, 1)
        self.mae_bns1 = nn.BatchNorm1d(512)
        self.mae_bns2 = nn.BatchNorm1d(256)

        self.mae_relu = nn.ReLU()

        

    def forward(self, xyz, cls_label):
        cls_label = cls_label.reshape(cls_label.shape[0], 16)
        norm = xyz[:, :, 3:]
        xyz = xyz[:, :, :3]
        B, N_original, _ = xyz.shape  # 保存原始点云的点数
        l0_xyz, l0_norm, l0_points, l0_points_pos = self.sa0(xyz, norm, None)
        l0_points_pos = l0_points_pos.permute(0, 2, 1)
        l0_points = l0_points.permute(0, 2, 1)
        # option1,这里可以输入xyz；option2，这里输入14维坐标
        group_input_tokens = l0_points
        
        
        # final input
        x = group_input_tokens
        pos = l0_points_pos
        B, N, C = x.shape  # N是采样后的点数（512）
        # transformer
        feature_list = self.mae_blocks(x, pos)
        feature_list = [self.mae_norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
        x_max = torch.max(x,2)[0]
        x_avg = torch.mean(x,2)
        # 使用原始点云的点数N_original来repeat，而不是采样后的点数N
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N_original)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N_original)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.mae_label_conv(cls_label_one_hot).repeat(1, 1, N_original)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1) #1152*2 + 64

        f_level_0 = self.mae_propagation_0(xyz.transpose(-1, -2), l0_xyz.transpose(-1, -2), xyz.transpose(-1, -2), x)

        x = torch.cat((f_level_0,x_global_feature), 1)
        x = self.mae_relu(self.mae_bns1(self.mae_convs1(x)))
        x = self.mae_dp1(x)
        x = self.mae_relu(self.mae_bns2(self.mae_convs2(x)))
        x = self.mae_convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, None

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        get_missing_parameters_message(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        get_unexpected_parameters_message(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

def group_index(nsample, radius, xyz, new_xyz, group='knn'):
    if group == 'knn':
        idx = knn_point(nsample, xyz, new_xyz.contiguous())


    elif group == 'ball':
        idx = pointops.ballquery(radius, nsample, xyz, new_xyz.contiguous())
        idx = idx.long()
    else:
        print('Unknown grouping method!')
        exit()

    return idx

