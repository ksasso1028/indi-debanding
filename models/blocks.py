import torch
import torch.nn as nn
from torch.nn.utils import weight_norm



class PaddedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dil=1, weight=True):
        super(PaddedConv2d, self).__init__()
        self.kernel = kernel_size
        self.d = dil
        self.weight = weight
        if self.weight:
            self.depth = weight_norm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size),
                          padding=((kernel_size - 1) // 2 * dil, (kernel_size - 1) // 2 * dil), stride=(stride),
                          dilation=(dil, dil)))
        else:
            self.depth = (
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size),
                          padding=((kernel_size - 1) // 2 * dil, (kernel_size - 1) // 2 * dil), stride=(stride),
                          dilation=(dil, dil)))
    def forward(self, x):
        x = self.depth(x)
        return x

class EncoderFFTime2d(nn.Module):
    def __init__(self, channels, dil=1, stride=1, mul=1, l=False, kernel=3,linear=32,weight=False):
        super(EncoderFFTime2d, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.conv1 = PaddedConv2d(channels, channels + mul, kernel_size=self.kernel, stride=stride, dil=dil,weight=weight)
        self.conv2 = PaddedConv2d(channels + mul, channels + mul, kernel_size=self.kernel, stride=1, dil=dil,weight=weight)
        self.conv3 = PaddedConv2d(channels + mul, channels + mul, kernel_size=self.kernel, stride=1, dil=dil,weight=weight)
        self.one = PaddedConv2d(channels + mul, channels + mul, kernel_size= 1)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                linear,
                channels + mul
            ),
        )
    def forward(self, x, t):
        # downsample layer
        x = torch.sin(self.conv1(x))
        emb = (self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]))
        x = x + emb
        # start block
        clone = x.clone()
        x =  torch.sin(self.conv2(x))
        x =  torch.sin(self.conv3(x))
        x = x + clone
        return x

class UpscaleFFTime2d(nn.Module):
    def __init__(self, channels, dil, stride=1, scale_factor=2,mul=1, kernel=3, linear=32, weight=False):
        super(UpscaleFFTime2d, self).__init__()
        self.stride = stride
        self.kernel = kernel
        self.conv1 = nn.Upsample(scale_factor=scale_factor)
        # self.conv1 = nn.ConvTranspose1d(channels, channels-mul, stride= 4,kernel_size=self.kernel, padding=0)
        self.conv2 = PaddedConv2d(channels, channels - mul, kernel_size=self.kernel, dil=dil, weight=weight)
        self.conv3 = PaddedConv2d(channels - mul, channels - mul, kernel_size=self.kernel, dil=dil,weight=weight)
        self.conv4 = PaddedConv2d(channels - mul, channels - mul, kernel_size=self.kernel, dil=dil,weight=weight)
        self.one = PaddedConv2d(channels-mul, channels-mul, kernel_size=1)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                linear,
                channels - mul
            ),
        )

    def forward(self, x, t):
        x = self.conv1(x)
        x =  torch.sin(self.conv2(x))
        emb = (self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]))
        x = x + emb
        # start block
        clone = x.clone()
        x =  torch.sin(self.conv3(x))
        x =  torch.sin(self.conv4(x))
        x = x + clone
        return x



def Normalize(in_channels, num_groups=8):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class Snake(nn.Module):
    def __init__(self, inFeatures):
        super(Snake, self).__init__()
        self.scale = nn.Parameter(torch.Tensor(1, inFeatures))
        nn.init.uniform_(self.scale, a=.1, b=3)  # Initialize scale parameter

    def forward(self, x):
        x = x.transpose(1,-1)
        x = x + (1.0/self.scale)* pow(torch.sin(x * self.scale),2)
        #x = torch.sin(x)
        #x = x * self.scale
        # x = torch.nn.functional.leaky_relu(x,.56)
        x = x.transpose(1,-1)
        return x