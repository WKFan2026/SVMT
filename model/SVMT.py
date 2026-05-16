#This section of the code is adapted from
# @inproceedings{ranftl2021vision,
#   title={Vision transformers for dense prediction},
#   author={Ranftl, Ren{\'e} and Bochkovskiy, Alexey and Koltun, Vladlen},
#   booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
#   pages={12179--12188},
#   year={2021}
# }

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model import DenseConvEncoder

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

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

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

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)
    
class Interpolate(nn.Module):

    def __init__(self, scale_factor, mode, align_corners=False):

        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

class ResidualConvUnit_custom(nn.Module):

    def __init__(self, features, activation, bn):
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)
    
class FeatureFusionBlock_custom(nn.Module):

    def __init__(
        self,
        features=256,
        feature_concat = None,
        activation=nn.ReLU(False),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):

        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        if feature_concat != None:
            self.feature_fusion_conv = nn.Conv2d(
                feature_concat,
                out_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
            )

    def forward(self, *xs):
        output = xs[0]

        if len(xs) >= 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        if len(xs) >2:
            fcde_feature = xs[2] 
            output = self.feature_fusion_conv(torch.cat((output,fcde_feature),1))           

        output = self.out_conv(output)
        
        return output

class Depth_decoder(nn.Module):

    def __init__(self,features,out_channel,non_negative = True):
        super().__init__()

        self.layer1_skip_process = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
                dilation=1,
                groups=1,
            ),
            nn.Conv2d(
                in_channels = features[0],
                out_channels = out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )
        )
        
        self.layer2_skip_process = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
                dilation=1,
                groups=1,
            ),
            nn.Conv2d(
                in_channels = features[1],
                out_channels = out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )
        )
        self.layer3_skip_process = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=features[2],
                out_channels=features[2],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
                dilation=1,
                groups=1,
            ),
            nn.Conv2d(
                in_channels = features[2],
                out_channels = out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )
        )
        self.layer4_skip_process = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
                dilation=1,
                groups=1,
            ),
            nn.Conv2d(
                in_channels = features[3],
                out_channels = out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )
        )
        
        self.fusion4 = FeatureFusionBlock_custom(features = out_channel)
        self.fusion3 = FeatureFusionBlock_custom(features = out_channel,feature_concat = out_channel+192)
        self.fusion2 = FeatureFusionBlock_custom(features = out_channel,feature_concat = out_channel+144)
        self.fusion1 = FeatureFusionBlock_custom(features = out_channel,feature_concat = out_channel+96)

        self.head = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(out_channel // 2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        
    def forward(self,layer1,layer2,layer3,layer4,Fcde):
        temp1 = self.layer1_skip_process(layer1)
        temp2 = self.layer2_skip_process(layer2)
        temp3 = self.layer3_skip_process(layer3)
        temp4 = self.layer4_skip_process(layer4)

        path_4 = self.fusion4(temp4)
        path_3 = self.fusion3(path_4,temp3,Fcde[3])
        path_2 = self.fusion2(path_3,temp2,Fcde[2])
        path_1 = self.fusion1(path_2,temp1,Fcde[1])

        depth_map = self.head(path_1)
        
        return depth_map,path_1
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
 
    def forward(self,input):
        return input.view(input.size(0),-1)

def vector_to_matrix(R):
    B = R.shape[0]

    theta = torch.norm(R, p=2, dim=1).unsqueeze(0).transpose(0, 1)
    n = R / theta

    x_1, y_1, z_1 = n[:, 0], n[:, 1], n[:, 2]
    zeros = z_1.detach() * 0

    inver_matrix = torch.stack([zeros, -z_1, y_1,
                                z_1, zeros, -x_1,
                                -y_1, x_1, zeros], dim=1).reshape(B, 3, 3)

    zeros = zeros.unsqueeze(0).transpose(0, 1)
    re_1 = torch.stack([torch.cos(theta), zeros,zeros,
                                zeros, torch.cos(theta), zeros,
                                zeros, zeros, torch.cos(theta)], dim=1).reshape(B, 3, 3)

    theta = theta.unsqueeze(2).repeat(1, 3, 3)
    re_2 = torch.mul(1 - torch.cos(theta), torch.matmul(n.unsqueeze(2), n.unsqueeze(2).transpose(1, 2)))
    re_3 = torch.mul(torch.sin(theta), inver_matrix)
    return re_1+re_2+re_3

class Pose_decoder(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
        self.obtain_pose = nn.Sequential(
            
            #1
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim//2,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            
            nn.BatchNorm2d(dim//2),
            nn.ReLU(True),
            #2
            nn.Conv2d(
                in_channels=dim//2,
                out_channels=dim//4,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            
            nn.BatchNorm2d(dim//4),
            nn.ReLU(True),
            #3
            nn.Conv2d(
                in_channels=dim//4,
                out_channels=dim//8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            
            nn.BatchNorm2d(dim//8),
            nn.ReLU(True),
            #4
            nn.Conv2d(
                in_channels=dim//8,
                out_channels=dim//16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            
            nn.BatchNorm2d(dim//16),
            nn.ReLU(True),
            
            Flatten(),
            
            nn.Linear(2560,1280),
            nn.ReLU(True),
            nn.Linear(1280,640),
            nn.ReLU(True),
            nn.Linear(640,320),
            nn.ReLU(True),
            nn.Linear(320,6),
            
        )
        
        
    def forward(self,x):
        x = self.obtain_pose(x)
        R = x[:,:3]
        T = x[:,3:]
        R_re = vector_to_matrix(R)
        T = T.unsqueeze(2)
        return R_re,T

class StructureValidMaskTransformer(nn.Module):

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dims=[384,768,768,768],
                 num_heads=[6, 12, 12, 12], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 3, 3], sr_ratios=[1, 1, 1, 1], num_stages=4, non_negative=True, out_channel = 256,
                 pose_mask_para_phi = 0.2):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.pose_mask_para_phi = pose_mask_para_phi

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        
        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

        #local feature
        self.fcd_encoder = DenseConvEncoder.FCDenseNet()

        #depth-decoder
        self.depth_decoder = Depth_decoder(embed_dims,out_channel,non_negative)
        
        #pose-decoder
        self.pose_decoder = Pose_decoder(out_channel*2)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear",align_corners=True).reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, imgs,sv_masks_model):
        
        B = imgs.shape[0]
        im_num = imgs.shape[1]
        
        #local feature extraction
        fcd_feature_all_im = []
        for i in range(im_num):
            fcd_feature_all_im.append(self.fcd_encoder(imgs[:,i,:,:,:]))
        
        #global feature extraction
        global_feature_all = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            
            for j in range(im_num):
                if j==0:
                    x, (H, W) = patch_embed(imgs[:,j,:,:,:])
                    pos_embed1 = self._get_pos_embed(pos_embed, patch_embed, H, W)
                    x = pos_drop(x + pos_embed1)
                else:
                    x_temp, (H, W) = patch_embed(imgs[:,j,:,:,:])
                    pos_embed_temp = self._get_pos_embed(pos_embed, patch_embed, H, W)
                    x_temp = pos_drop(x_temp + pos_embed_temp)
                    x = torch.cat((x,x_temp),1)

            for blk in block:
                x = blk(x)

            tn = int(H*W)
            for j in range(im_num):
                if j==0:
                    imgs = x[:,tn*j:tn*(j+1),:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                    imgs = imgs.unsqueeze(1)
                else:
                    imgs_temp = x[:,tn*j:tn*(j+1),:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                    imgs_temp = imgs_temp.unsqueeze(1)
                    imgs = torch.cat((imgs,imgs_temp),1)
            
            
            global_feature_all.append(imgs)
        
        
        #decoder-feature fusion
        
        for i in range(im_num):
            if i==0:
                depth_map,feature_lg = self.depth_decoder(global_feature_all[0][:,i,:,:,:],
                                               global_feature_all[1][:,i,:,:,:],
                                               global_feature_all[2][:,i,:,:,:],
                                               global_feature_all[3][:,i,:,:,:],
                                              fcd_feature_all_im[i])

                depth_map = torch.clamp(depth_map, min=1e-8)
                depth_map = 1.0 / depth_map
                
                feature_lg = feature_lg.unsqueeze(1)

            else:
                depth_map_temp,feature_lg_temp = self.depth_decoder(global_feature_all[0][:,i,:,:,:],
                                               global_feature_all[1][:,i,:,:,:],
                                               global_feature_all[2][:,i,:,:,:],
                                               global_feature_all[3][:,i,:,:,:],
                                              fcd_feature_all_im[i])

                depth_map_temp = torch.clamp(depth_map_temp, min=1e-8)
                depth_map_temp = 1.0 / depth_map_temp

                depth_map = torch.cat((depth_map,depth_map_temp),1)
                feature_lg_temp = feature_lg_temp.unsqueeze(1)
                feature_lg = torch.cat((feature_lg,feature_lg_temp),1)
        
        
        #pose decoder
        for i in range(im_num-1):
            
            input1 = feature_lg[:,i,:,:,:]
            input2 = feature_lg[:,i+1,:,:,:]
            
            input_features12 = torch.cat((input1,input2),1)
            input_features12 = input_features12*(1+self.pose_mask_para_phi*sv_masks_model[:,i,:,:,:])

            if i==0:
                all_R12,all_T12 = self.pose_decoder(input_features12)
                all_R12 = all_R12.unsqueeze(1)
                all_T12 = all_T12.unsqueeze(1)

            else:
                temp_R12,temp_T12 = self.pose_decoder(input_features12)
                all_R12 = torch.cat((all_R12,temp_R12.unsqueeze(1)),1)
                all_T12 = torch.cat((all_T12,temp_T12.unsqueeze(1)),1)
        
        
        return depth_map,all_R12,all_T12


