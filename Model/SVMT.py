import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(value):
    """Convert a scalar image/patch size to the (height, width) form used here."""
    return value if isinstance(value, tuple) else (value, value)


class DropPath(nn.Module):
    """Per-sample stochastic depth, equivalent to timm's default DropPath."""

    def __init__(self, drop_prob=0., scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0. and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module(
            'conv',
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True),
        )


class DenseConvBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)
        ])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)

        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True))
        self.add_module('maxpool', nn.MaxPool2d(2))


class DenseConvBranch(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(4, 4, 4), growth_rate=12, out_chans_first_conv=48):
        super().__init__()
        if len(down_blocks) != 3:
            raise ValueError('SVMT depth decoder requires exactly three DenseNet down blocks.')

        self.down_blocks = tuple(down_blocks)
        self.firstconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_chans_first_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        cur_channels_count = out_chans_first_conv
        feature_channels = [cur_channels_count]

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for n_layers in self.down_blocks:
            self.denseBlocksDown.append(DenseConvBlock(cur_channels_count, growth_rate, n_layers))
            cur_channels_count += growth_rate * n_layers
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
            feature_channels.append(cur_channels_count)

        self.feature_channels = tuple(feature_channels)

    def forward(self, x):
        skip_connections = []
        out = self.firstconv(x)
        skip_connections.append(out)

        for dense_block, transition_down in zip(self.denseBlocksDown, self.transDownBlocks):
            out = dense_block(out)
            out = transition_down(out)
            skip_connections.append(out)

        return skip_connections

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


class HiTransBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
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

class ResBlock(nn.Module):

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
    
class ResAtteFusion(nn.Module):

    def __init__(
        self,
        features=256,
        feature_concat = None,
        activation=nn.ReLU(False),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        upsample_scale_factor=2,
        upsample_mode="bilinear",
    ):

        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.upsample_scale_factor = upsample_scale_factor
        self.upsample_mode = upsample_mode
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

        self.resConfUnit1 = ResBlock(features, activation, bn)
        self.resConfUnit2 = ResBlock(features, activation, bn)

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
            output,
            scale_factor=self.upsample_scale_factor,
            mode=self.upsample_mode,
            align_corners=self.align_corners,
        )

        if len(xs) >2:
            fcde_feature = xs[2] 
            output = self.feature_fusion_conv(torch.cat((output,fcde_feature),1))           

        output = self.out_conv(output)
        
        return output

class Depth_decoder(nn.Module):

    def __init__(
        self,
        features,
        out_channel,
        local_feature_channels,
        non_negative=True,
        skip_upsample_kernel_size=2,
        skip_upsample_stride=2,
        skip_conv_kernel_size=3,
        fusion_activation=None,
        fusion_batch_norm=False,
        fusion_align_corners=True,
        fusion_upsample_scale_factor=2,
        fusion_upsample_mode="bilinear",
        head_hidden_channels=64,
        head_upsample_scale_factor=2,
        head_upsample_mode="bilinear",
        head_align_corners=True,
    ):
        super().__init__()
        if len(features) != 4:
            raise ValueError('Depth_decoder requires four transformer feature-channel values.')
        if len(local_feature_channels) != 4:
            raise ValueError('Depth_decoder requires four DenseNet feature-channel values.')

        fusion_activation = fusion_activation if fusion_activation is not None else nn.ReLU(False)
        skip_conv_padding = skip_conv_kernel_size // 2
        fusion_kwargs = dict(
            activation=fusion_activation,
            bn=fusion_batch_norm,
            align_corners=fusion_align_corners,
            upsample_scale_factor=fusion_upsample_scale_factor,
            upsample_mode=fusion_upsample_mode,
        )

        self.layer1_skip_process = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=skip_upsample_kernel_size,
                stride=skip_upsample_stride,
                padding=0,
                bias=False,
                dilation=1,
                groups=1,
            ),
            nn.Conv2d(
                in_channels = features[0],
                out_channels = out_channel,
                kernel_size=skip_conv_kernel_size,
                stride=1,
                padding=skip_conv_padding,
                bias=False,
                groups=1,
            )
        )
        
        self.layer2_skip_process = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=skip_upsample_kernel_size,
                stride=skip_upsample_stride,
                padding=0,
                bias=False,
                dilation=1,
                groups=1,
            ),
            nn.Conv2d(
                in_channels = features[1],
                out_channels = out_channel,
                kernel_size=skip_conv_kernel_size,
                stride=1,
                padding=skip_conv_padding,
                bias=False,
                groups=1,
            )
        )
        self.layer3_skip_process = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=features[2],
                out_channels=features[2],
                kernel_size=skip_upsample_kernel_size,
                stride=skip_upsample_stride,
                padding=0,
                bias=False,
                dilation=1,
                groups=1,
            ),
            nn.Conv2d(
                in_channels = features[2],
                out_channels = out_channel,
                kernel_size=skip_conv_kernel_size,
                stride=1,
                padding=skip_conv_padding,
                bias=False,
                groups=1,
            )
        )
        self.layer4_skip_process = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=skip_upsample_kernel_size,
                stride=skip_upsample_stride,
                padding=0,
                bias=False,
                dilation=1,
                groups=1,
            ),
            nn.Conv2d(
                in_channels = features[3],
                out_channels = out_channel,
                kernel_size=skip_conv_kernel_size,
                stride=1,
                padding=skip_conv_padding,
                bias=False,
                groups=1,
            )
        )
        
        self.fusion4 = ResAtteFusion(features=out_channel, **fusion_kwargs)
        self.fusion3 = ResAtteFusion(
            features=out_channel, feature_concat=out_channel + local_feature_channels[3], **fusion_kwargs
        )
        self.fusion2 = ResAtteFusion(
            features=out_channel, feature_concat=out_channel + local_feature_channels[2], **fusion_kwargs
        )
        self.fusion1 = ResAtteFusion(
            features=out_channel, feature_concat=out_channel + local_feature_channels[1], **fusion_kwargs
        )

        self.head = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(
                scale_factor=head_upsample_scale_factor,
                mode=head_upsample_mode,
                align_corners=head_align_corners,
            ),
            nn.Conv2d(out_channel // 2, head_hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(head_hidden_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus() if non_negative else nn.Identity(),
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
    theta = torch.linalg.vector_norm(R, dim=1, keepdim=True).view(B, 1, 1)
    x_1, y_1, z_1 = R.unbind(dim=1)
    zeros = torch.zeros_like(x_1)
    skew_matrix = torch.stack([
        zeros, -z_1, y_1,
        z_1, zeros, -x_1,
        -y_1, x_1, zeros,
    ], dim=1).reshape(B, 3, 3)
    identity = torch.eye(3, dtype=R.dtype, device=R.device).unsqueeze(0).expand(B, -1, -1)
    first_coefficient = torch.sinc(theta / torch.pi)
    second_coefficient = 0.5 * torch.sinc(theta / (2 * torch.pi)).square()
    return identity + first_coefficient * skew_matrix + second_coefficient * (skew_matrix @ skew_matrix)

class Pose_decoder(nn.Module):
    def __init__(
        self,
        dim,
        conv_channels=None,
        conv_kernel_sizes=(5, 5, 3, 3),
        conv_strides=(2, 2, 2, 2),
        conv_paddings=(2, 2, 1, 1),
        pooled_size=(8, 10),
        fc_features=(1280, 640, 320, 6),
        initialize_to_identity=True,
    ):
        super().__init__()
        conv_channels = tuple(conv_channels or (dim // 2, dim // 4, dim // 8, dim // 16))
        conv_kernel_sizes = tuple(conv_kernel_sizes)
        conv_strides = tuple(conv_strides)
        conv_paddings = tuple(conv_paddings)
        fc_features = tuple(fc_features)
        if not (len(conv_channels) == len(conv_kernel_sizes) == len(conv_strides) == len(conv_paddings) == 4):
            raise ValueError('Pose_decoder requires four convolution-stage specifications.')
        if len(fc_features) != 4 or fc_features[-1] != 6:
            raise ValueError('Pose_decoder fc_features must contain three hidden sizes followed by output size 6.')

        # Pooling makes the pose MLP independent of the input image resolution.
        # For the default 256 x 320 input its 8 x 10 input is unchanged.
        self.pool = nn.AdaptiveAvgPool2d(pooled_size) if pooled_size is not None else nn.Identity()
        self.conv_stage_module_count = len(conv_channels) * 3
        pooled_area = 1 if pooled_size is None else pooled_size[0] * pooled_size[1]
        fc_input_features = conv_channels[-1] * pooled_area
        layers = []
        in_channels = dim
        for out_channels, kernel_size, stride, padding in zip(
            conv_channels, conv_kernel_sizes, conv_strides, conv_paddings
        ):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            ])
            in_channels = out_channels
        layers.append(Flatten())
        in_features = fc_input_features
        for out_features in fc_features:
            layers.append(nn.Linear(in_features, out_features))
            if out_features != fc_features[-1]:
                layers.append(nn.ReLU(True))
            in_features = out_features
        self.obtain_pose = nn.Sequential(*layers)
        if initialize_to_identity:
            nn.init.zeros_(self.obtain_pose[-1].weight)
            nn.init.zeros_(self.obtain_pose[-1].bias)

    def forward(self,x):
        x = self.obtain_pose[:self.conv_stage_module_count](x)
        x = self.pool(x)
        x = self.obtain_pose[self.conv_stage_module_count:](x)
        R = x[:,:3]
        T = x[:,3:]
        R_re = vector_to_matrix(R)
        T = T.unsqueeze(2)
        return R_re,T

class StructureValidMaskTransformer(nn.Module):

    def __init__(
        self,
        img_size_h=256,  # Fixed parameter
        img_size_w=320,  # Fixed parameter
        patch_size=8,  # Fixed parameter
        in_chans=3,
        embed_dims=(192, 384, 576, 768),
        num_heads=(3, 6, 9, 12),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        depths=(3, 3, 3, 3),
        sr_ratios=(1, 1, 1, 1),
        num_stages=4,  # Fixed parameter
        non_negative=True,
        inverse_depth_epsilon=1e-6,
        out_channel=256,
        pose_mask_para_phi=0.2,
        initialize_pose_to_identity=True,
        dense_down_blocks=(4, 4, 4),
        dense_growth_rate=12,
        dense_first_conv_channels=48,
        decoder_skip_upsample_kernel_size=2,
        decoder_skip_upsample_stride=2,
        decoder_skip_conv_kernel_size=3,
        decoder_fusion_activation=None,
        decoder_fusion_batch_norm=False,
        decoder_fusion_align_corners=True,
        decoder_fusion_upsample_scale_factor=2,
        decoder_fusion_upsample_mode="bilinear",
        decoder_head_hidden_channels=64,
        decoder_head_upsample_scale_factor=2,
        decoder_head_upsample_mode="bilinear",
        decoder_head_align_corners=True,
        pose_conv_channels=None,
        pose_conv_kernel_sizes=(5, 5, 3, 3),
        pose_conv_strides=(2, 2, 2, 2),
        pose_conv_paddings=(2, 2, 1, 1),
        pose_pooled_size=(8, 10),  # Fixed parameter
        pose_fc_features=(1280, 640, 320, 6),  # Fixed parameter
    ):
        super().__init__()
        if num_stages != 4:
            raise ValueError('SVMT currently requires four transformer stages for its depth decoder.')
        if not all(len(values) == num_stages for values in (embed_dims, num_heads, mlp_ratios, depths, sr_ratios)):
            raise ValueError('embed_dims, num_heads, mlp_ratios, depths, and sr_ratios must match num_stages.')
        if inverse_depth_epsilon <= 0:
            raise ValueError('inverse_depth_epsilon must be positive.')
        if isinstance(norm_layer, str):
            if not hasattr(nn, norm_layer):
                raise ValueError(f"Unsupported norm_layer: {norm_layer}.")
            norm_layer = getattr(nn, norm_layer)

        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.img_size = (img_size_h, img_size_w)
        self.in_chans = in_chans
        self.embed_dims = tuple(embed_dims)
        self.num_heads = tuple(num_heads)
        self.mlp_ratios = tuple(mlp_ratios)
        self.depths = tuple(depths)
        self.sr_ratios = tuple(sr_ratios)
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.pose_mask_para_phi = pose_mask_para_phi
        self.inverse_depth_epsilon = inverse_depth_epsilon
        self.dense_down_blocks = tuple(dense_down_blocks)
        self.dense_growth_rate = dense_growth_rate
        self.dense_first_conv_channels = dense_first_conv_channels
        self.out_channel = out_channel
        self.decoder_config = {
            'skip_upsample_kernel_size': decoder_skip_upsample_kernel_size,
            'skip_upsample_stride': decoder_skip_upsample_stride,
            'skip_conv_kernel_size': decoder_skip_conv_kernel_size,
            'fusion_activation': decoder_fusion_activation,
            'fusion_batch_norm': decoder_fusion_batch_norm,
            'fusion_align_corners': decoder_fusion_align_corners,
            'fusion_upsample_scale_factor': decoder_fusion_upsample_scale_factor,
            'fusion_upsample_mode': decoder_fusion_upsample_mode,
            'head_hidden_channels': decoder_head_hidden_channels,
            'head_upsample_scale_factor': decoder_head_upsample_scale_factor,
            'head_upsample_mode': decoder_head_upsample_mode,
            'head_align_corners': decoder_head_align_corners,
        }
        self.pose_config = {
            'conv_channels': None if pose_conv_channels is None else tuple(pose_conv_channels),
            'conv_kernel_sizes': tuple(pose_conv_kernel_sizes),
            'conv_strides': tuple(pose_conv_strides),
            'conv_paddings': tuple(pose_conv_paddings),
            'pooled_size': pose_pooled_size,
            'fc_features': tuple(pose_fc_features),
        }

        initial_patch_size = to_2tuple(patch_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(num_stages):
            if i == 0:
                stage_img_size = self.img_size
                stage_patch_size = initial_patch_size
            else:
                # Each preceding transformer stage reconstructs a feature map
                # at 1 / patch_size, then every following stage downsamples by 2.
                stage_img_size = (
                    img_size_h // (initial_patch_size[0] * (2 ** (i - 1))),
                    img_size_w // (initial_patch_size[1] * (2 ** (i - 1))),
                )
                stage_patch_size = (2, 2)

            patch_embed = PatchEmbed(img_size=stage_img_size,
                                     patch_size=stage_patch_size,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([HiTransBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            nn.init.trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

        self.fcd_encoder = DenseConvBranch(
            in_channels=in_chans,
            down_blocks=self.dense_down_blocks,
            growth_rate=dense_growth_rate,
            out_chans_first_conv=dense_first_conv_channels,
        )

        self.depth_decoder = Depth_decoder(
            self.embed_dims,
            out_channel,
            self.fcd_encoder.feature_channels,
            non_negative=non_negative,
            **self.decoder_config,
        )
        
        self.pose_decoder = Pose_decoder(
            out_channel * 2,
            initialize_to_identity=initialize_pose_to_identity,
            **self.pose_config,
        )


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H == patch_embed.H and W == patch_embed.W:
            return pos_embed
        return F.interpolate(
            pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear", align_corners=True
        ).reshape(1, -1, H * W).permute(0, 2, 1)

    def forward(self, imgs, sv_masks_model=None):
        if not isinstance(imgs, torch.Tensor):
            raise TypeError("imgs must be a torch.Tensor with shape [B, N, C, H, W].")
        if imgs.ndim != 5:
            raise ValueError(
                f"imgs must have shape [B, N, C, H, W], but received a {imgs.ndim}D tensor with shape {tuple(imgs.shape)}."
            )

        B = imgs.shape[0]
        im_num = imgs.shape[1]
        channels = imgs.shape[2]
        original_size = imgs.shape[-2:]
        if im_num <= 2:
            raise ValueError(f"imgs must contain more than two frames (N > 2), but received N={im_num}.")
        if channels != self.in_chans:
            raise ValueError(f"imgs has C={channels} channels, but this model expects in_chans={self.in_chans}.")

        # Run the network at its configured image resolution and restore the
        # predicted depth maps to the input resolution before returning them.
        if original_size != self.img_size:
            imgs = F.interpolate(
                imgs.reshape(B * im_num, channels, *original_size),
                size=self.img_size,
                mode="bilinear",
                align_corners=False,
            ).reshape(B, im_num, channels, *self.img_size)

        if sv_masks_model is not None:
            if not isinstance(sv_masks_model, torch.Tensor) or sv_masks_model.ndim != 5:
                raise ValueError("sv_masks_model must have shape [B, N - 1, 1, H, W] when provided.")
            expected_mask_shape = (B, im_num - 1, 1)
            if tuple(sv_masks_model.shape[:3]) != expected_mask_shape:
                raise ValueError(
                    "sv_masks_model must have shape "
                    f"[{B}, {im_num - 1}, 1, H, W], but received {tuple(sv_masks_model.shape)}."
                )

            pose_feature_size = (self.img_size_h // 2, self.img_size_w // 2)
            if tuple(sv_masks_model.shape[-2:]) != pose_feature_size:
                sv_masks_model = F.interpolate(
                    sv_masks_model,
                    size=(1, *pose_feature_size),
                    mode="nearest",
                )

        # Keep each image as an individual encoder call.  This preserves the
        # original BatchNorm and dropout behaviour during training.
        fcd_feature_all_im = []
        for i in range(im_num):
            fcd_feature_all_im.append(self.fcd_encoder(imgs[:,i,:,:,:]))

        # Global feature extraction.  `current_images` is the stage input;
        # collecting token/image tensors first avoids repeated growing cat
        # allocations without changing their order or values.
        global_feature_all = []
        current_images = imgs
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")

            image_tokens = []
            for j in range(im_num):
                x_temp, (H, W) = patch_embed(current_images[:,j,:,:,:])
                pos_embed_temp = self._get_pos_embed(pos_embed, patch_embed, H, W)
                image_tokens.append(pos_drop(x_temp + pos_embed_temp))
            x = torch.cat(image_tokens, dim=1)

            for blk in block:
                x = blk(x)

            tokens_per_image = int(H * W)
            stage_images = [
                x[:, tokens_per_image * j:tokens_per_image * (j + 1), :]
                .reshape(B, H, W, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for j in range(im_num)
            ]
            current_images = torch.stack(stage_images, dim=1)
            global_feature_all.append(current_images)

        # Decoder-feature fusion.  Decoder calls remain in their original
        # image order; only the final output assembly is batched.
        depth_maps = []
        local_global_features = []
        for i in range(im_num):
            depth_map_temp, feature_lg_temp = self.depth_decoder(
                global_feature_all[0][:,i,:,:,:],
                global_feature_all[1][:,i,:,:,:],
                global_feature_all[2][:,i,:,:,:],
                global_feature_all[3][:,i,:,:,:],
                fcd_feature_all_im[i],
            )
            depth_maps.append(torch.reciprocal(depth_map_temp + self.inverse_depth_epsilon))
            local_global_features.append(feature_lg_temp)
        depth_map = torch.cat(depth_maps, dim=1)
        feature_lg = torch.stack(local_global_features, dim=1)

        # Pose decoder: retain the original pairwise call order and assemble
        # its outputs once to avoid repeated growing cat allocations.
        rotations = []
        translations = []
        for i in range(im_num-1):
            input1 = feature_lg[:,i,:,:,:]
            input2 = feature_lg[:,i+1,:,:,:]
            input_features12 = torch.cat((input1,input2), 1)
            if sv_masks_model is not None:
                input_features12 = input_features12 * (
                    1 + self.pose_mask_para_phi * sv_masks_model[:,i,:,:,:]
                )
            rotation, translation = self.pose_decoder(input_features12)
            rotations.append(rotation)
            translations.append(translation)

        all_R12 = torch.stack(rotations, dim=1)
        all_T12 = torch.stack(translations, dim=1)
        if original_size != self.img_size:
            depth_map = F.interpolate(
                depth_map,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return depth_map,all_R12,all_T12
