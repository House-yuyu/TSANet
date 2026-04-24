import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.TCA import *
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCA(nn.Module):
    def __init__(self, inp,  kernel_size=1, ratio=1, band_kernel_size=11,dw_size=(1,1), padding=(0,0), stride=1, square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size//2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc=inp//ratio
        self.excite = nn.Sequential(
                nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc),
                nn.Sigmoid()
            )
    
    def sge(self, x):
        #[N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w #.repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather) # [N, 1, C, 1]
        
        return ge

    def forward(self, x):
        loc=self.dwconv_hw(x)
        att=self.sge(x)
        out = att*loc
        
        return out

class RCM(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=RCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size, ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.avg_pool = nn.AdaptiveMaxPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [Conv(n_feat, n_feat, kernel_size, bias=bias), act, Conv(n_feat, n_feat, kernel_size, bias=bias)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, stride=1, norm=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
        self.norm = norm

        if self.norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif self.norm == 'LN':
            self.norm_layer = nn.InstanceNorm2d(num_features=out_channels, momentum=0.3, affine=True, track_running_stats=True)

    def forward(self, x):
        if self.norm:
            return self.norm_layer(self.conv(x))
        else:
            return self.conv(x)

class PGB(nn.Module):
    def __init__(self, in_channel=3, f_channel=64, g_channel=1):
        super(PGB, self).__init__()
        self.conv1 = Conv(g_channel, 1, 1)
        self.conv2 = Conv(g_channel, f_channel, 1)
        self.conv3 = Conv(in_channel, f_channel, 3)
        self.conv4 = Conv(f_channel, f_channel, 3)
        self.conv5 = Conv(f_channel, f_channel, 3)

    def forward(self, img, guide_f):
        guide_mul = torch.sigmoid(self.conv1(guide_f))
        guide_add = self.conv2(guide_f)
        x = self.conv3(img)
        x = x * guide_mul
        x = self.conv4(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) 
        x = x + guide_add
        x = self.conv5(x)
        return x
    
class IEN(nn.Module):
    def __init__(self, in_channel=3, f_channel=48, w_channel=48):
        super(IEN, self).__init__()

        self.layer0 = nn.Sequential(
                      Conv(in_channel, f_channel, 3),
                      Conv(f_channel, f_channel, 3)
        )

        self.mu_conv = nn.Conv2d(w_channel, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.mu_conv.weight, 0.0)
        if self.mu_conv.bias is not None:
            nn.init.constant_(self.mu_conv.bias, 0.0)
        #self.para = torch.nn.Parameter(torch.ones(w_channel, 1, 1))

        self.guide1 = PGB(f_channel, f_channel, w_channel)
        self.layer1 = RCM(dim = f_channel,square_kernel_size=3)
        self.guide2 = PGB(f_channel, f_channel, w_channel)
        self.layer2 = RCM(dim = f_channel,square_kernel_size=3)
        self.guide3 = PGB(f_channel, f_channel, w_channel)
        self.layer3 = RCM(dim = f_channel,square_kernel_size=3)
        self.guide4 = PGB(f_channel, f_channel, w_channel)
        self.layer4 = RCM(dim = f_channel,square_kernel_size=3)
        self.guide5 = PGB(f_channel, f_channel, w_channel)
        self.layer5 = RCM(dim = f_channel,square_kernel_size=3)
        self.guide6 = PGB(f_channel, f_channel, w_channel)
        self.layer6 = RCM(dim = f_channel,square_kernel_size=3)
        self.out = Conv(f_channel, 3, 3)

    def forward(self, img, illu, rest):
        x = self.layer0(img)
        mu_map = 2.0 * torch.sigmoid(self.mu_conv(rest))
        res_illu = illu - mu_map * rest
        # res_illu = illu-self.para*rest
        x = self.guide1(x, res_illu)
        x = self.layer1(x)
        x = self.guide2(x, rest)
        x = self.layer2(x)
        x = self.guide3(x, res_illu)
        x = self.layer3(x)
        x = self.guide4(x, rest)
        x = self.layer4(x)
        x = self.guide5(x, res_illu)
        x = self.layer5(x)
        x = self.guide6(x, rest)
        x = self.layer6(x)
        x = self.out(x)

        return x


# class RCAB(nn.Module):
#     def __init__(self, in_feat, out_feat, kernel_size, reduction, n_blocks, bias=False, act=nn.ReLU(True)):
#         super(RCAB, self).__init__()
#         self.conv1 = Conv(in_feat, out_feat, 3)
#         self.cab = nn.Sequential(*[CAB(out_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=act) for _ in range(n_blocks)])
#         self.conv2 = Conv(out_feat, out_feat, 3)

#     def forward(self, x):
#         x = self.conv1(x)
#         x1 = self.cab(x) + x
#         x = self.conv2(x1)
#         return x

class PhaseGuidedChannelAttention(nn.Module):
    def __init__(self, channels, reduction=1):
        super().__init__()
        inter_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, phase_feature):
        y = self.avg_pool(phase_feature)  # [B, C, 1, 1]
        y = self.fc(y)                    # [B, C, 1, 1]
        return y

class FreBlock(nn.Module):
    def __init__(self, nc, out_nc=36):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.phase_proj = nn.Conv2d(nc, nc, 3, 1, 1)
        self.phase_attention = PhaseGuidedChannelAttention(nc)
        self.out_conv = nn.Conv2d(nc, out_nc, 1, 1, 0)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag_processed  = self.process1(mag)
        # pha_feature = self.process2(pha)
        pha_feature = self.phase_proj(pha)
        attention_weights = self.phase_attention(pha_feature)
        final_mag = mag_processed * attention_weights#  后续加残差
        real = final_mag * torch.cos(pha_feature)
        imag = final_mag * torch.sin(pha_feature)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return self.out_conv(x_out + x)
    
    
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        # 完全禁用HIN当输出通道数<2
        self.use_HIN = use_HIN if out_size >= 2 else False
        if self.use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):

        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out
    



class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # 存储初始参数
        self.initial_channel_num = channel_num
        self.initial_channel_split_num = channel_split_num
        
        self.clamp = clamp
        self.proj = nn.Conv2d(channel_num, 3, 1, 1, 0)
        self.F = None
        self.G = None
        self.H = None
        
    def forward(self, x):
        # 应用投影
        x = self.proj(x)
        proj_channels = x.size(1)  # 获取投影后的通道数
        split_len1 = max(1, proj_channels // 2)  # 至少为1
        
        split_len2 = proj_channels - split_len1   # 剩余通道

        if self.F is None:
            dev = x.device
            self.F = UNetConvBlock(split_len2, split_len1).to(dev)
            self.G = UNetConvBlock(split_len1, split_len2).to(dev)
            self.H = UNetConvBlock(split_len1, split_len2).to(dev)
        
        # 分割通道
        x1, x2 = (x.narrow(1, 0, split_len1), x.narrow(1, split_len1, split_len2))
        x2 = self.F(x2)
        # 后续处理
        y1 = x1 + x2
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        out = torch.cat((y1, y2), 1)
        
        return out
    
class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        channel_split_num = max(1,nc // 2)
        self.block = InvBlock(nc,channel_split_num)

    def forward(self, x):
        yy=self.block(x)

        return x+yy

class ProcessBlock(nn.Module):
    def __init__(self, in_nc,out_nc=1):
        super(ProcessBlock,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlock(in_nc,out_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.cat = nn.Conv2d(4,out_nc,1,1,0)

    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x = self.spatial_process(x)   # [B, 3, H, W]
        x_freq = self.frequency_process(xori)  # [B, 1, H, W]

        xcat = torch.cat([x,x_freq],1)  # [B, 4, H, W]
        x_out = self.cat(xcat)

        return x_out+xori
    

class TSANet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(TSANet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        self.i_four = FreBlock(nc=36,out_nc = 36)
        self.ien = IEN(3,64,64)

        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 64, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 64, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_TCA1 = HV_TCA(ch2, head2)
        self.HV_TCA2 = HV_TCA(ch3, head3)
        self.HV_TCA3 = HV_TCA(ch4, head4)
        self.HV_TCA4 = HV_TCA(ch4, head4)
        self.HV_TCA5 = HV_TCA(ch3, head3)
        self.HV_TCA6 = HV_TCA(ch2, head2)
        
        self.I_TCA1 = I_TCA(ch2, head2)
        self.I_TCA2 = I_TCA(ch3, head3)
        self.I_TCA3 = I_TCA(ch4, head4)
        self.I_TCA4 = I_TCA(ch4, head4)
        self.I_TCA5 = I_TCA(ch3, head3)
        self.I_TCA6 = I_TCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)   
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)  
        
        # low
        i_enc0 = self.IE_block0(i) 
        i_enc0 = self.i_four(i_enc0)
        i_enc1 = self.IE_block1(i_enc0) 
        hv_0 = self.HVE_block0(hvi) 
        hv_1 = self.HVE_block1(hv_0) 
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_TCA1(i_enc1, hv_1)
        hv_2 = self.HV_TCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_TCA2(i_enc2, hv_2)
        hv_3 = self.HV_TCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        i_enc4 = self.I_TCA3(i_enc3, hv_3)
        hv_4 = self.HV_TCA3(hv_3, i_enc3)
        
        i_dec4 = self.I_TCA4(i_enc4,hv_4)
        hv_4 = self.HV_TCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_TCA5(i_dec3, hv_3)
        hv_2 = self.HV_TCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_TCA6(i_dec2, hv_2)
        hv_1 = self.HV_TCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi =  self.ien(hvi,i_dec0,hv_0)
        
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
    





