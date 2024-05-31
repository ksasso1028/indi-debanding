import torch
import torch.nn as nn
from .blocks  import EncoderFFTime2d, UpscaleFFTime2d, PaddedConv2d,  AttnBlock, Snake
# 2D Unet that takes in a timestep T in encoder and decoder blocks
# uses sin activations, denoising target is done in the fourier domain instead of raw signal

class AutoFFTime2d(nn.Module):
    def __init__(self,
                 blocks,
                 in_channels,
                 kernel=3,
                 channel_factor=48,
                 layout = 2, # stereo, can be used for images just set spectrogram to false.
                 scale_factor = 2,
                 encoder_dil=4,
                 decoder_dil=1,
                 neck=False,
                 weight=False,
                 ):
        super(AutoFFTime2d, self).__init__()
        start = in_channels
        self.neck = neck
        self.scale_factor = scale_factor
        self.layout = layout
        self.c_factor = channel_factor
        self.linear = nn.Linear(1, in_channels)
        self.layout = layout
        self.conv = PaddedConv2d(self.layout, in_channels, kernel_size=3, stride=1)
        self.act = nn.SiLU()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.e_dil = encoder_dil
        self.d_dil = decoder_dil
        for block in range(blocks):
            stride = self.scale_factor
            self.encoder.append(
                EncoderFFTime2d(start, dil=1,stride=stride, mul=self.c_factor, weight=weight,linear=in_channels, kernel=kernel))
            # set channel size for next block
            start = start + self.c_factor
        # condition to add bottleneck
        self.linearAct = Snake(start)
        if self.neck:
            self.bottleneck = AttnBlock(start)
            # set channel size for next block
        for block in range(blocks):
            stride = 1
            self.decoder.append(
                UpscaleFFTime2d(start,dil=1, stride=stride, scale_factor = self.scale_factor, mul=self.c_factor, weight=weight,linear=in_channels, kernel=kernel))
            # set channel size for next block
            start = start - self.c_factor
        self.process = nn.Conv2d(start, self.layout, kernel_size=1, stride=1, padding=0)

    def forward(self, mix, t):
        t = (self.linear(t.unsqueeze(-1).type(torch.float).to(mix.device)))
        x = mix.clone()
        original = x[:, :self.layout, :].clone()
        x = self.act(self.conv(x))
        og = x.clone()
        features = []
        features.append(x)
        for module in self.encoder:
            x = module(x, t)
            features.append(x)
        clone = x.clone()
        if self.neck:
            x= self.bottleneck(x.clone())
            x = clone + x
        for i, module in enumerate(self.decoder):
            index = i + 1
            x = x[:, :, :features[-abs(index)].size(-2), :features[-abs(index)].size(-1)] + features[-abs(index)]
            x = module(x, t)
        x = x[:, :, :original.size(-2), :original.size(-1)]
        x = (og[:, :, :original.size(-2), :original.size(-1)] + x)
        # split X
        # layer specific resnet
        cut = self.process(x)
        synth = (cut[:, :self.layout, :original.size(-2), :original.size(-1)])
        out = synth
        return out

