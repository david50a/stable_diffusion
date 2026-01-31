import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
class VAE_AttentionBlock(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.groupnorm=nn.GroupNorm(32,channels)
        self.attention=SelfAttention(1,channels)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: (batch_size,channels,height,width)
        residue=x
        b,c,h,w=x.shape
        # x: (batch_size, channels, height, width)-> x: (batch_size,channels,height*width)
        x=x.view(b,c,h*w)
        # x: (batch_size,channels,height*width)-> x: (batch_size,height*width,channels)
        x=x.transpose(-1,-2)
        # x: (batch_size,channels,height*width)-> x: (batch_size,height*width,channels)
        self.attention(x)
        # x: (batch_size,channels,height*width)-> x: (batch_size,height*width,channels)
        x = x.transpose(-1, -2)
        # x: (batch_size,channels,height*width)-> (batch_size,channels,height,width)
        x=x.view(b,c,h,w)
        x+=residue
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.groupnorm_1=nn.GroupNorm(32,in_channel)
        self.conv_1=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1)
        self.groupnorm_2=nn.GroupNorm(32,in_channel)
        self.conv_2=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1)
        if in_channel==out_channel:
            self.residual_layer=nn.Identity()
        else:
            self.residual_layer=nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: (batch_size,in_channel,height,width)
        residue=x
        x=self.groupnorm_1(x)
        x=F.silu(x)
        x=self.conv_1(x)
        x=self.groupnorm_2(x)
        x=F.selu(x)
        x=self.conv_2(x)
        return x+self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            #(BATCH_SIZE,512,HEIGHT/8,WIDTH/8)-> (BATCH_SIZE,512,HEIGHT/8,WIDTH/8)
            VAE_ResidualBlock(512,512),
            #(BATCH_SIZE,512,HEIGHT/8,WIDTH/8)-> (BATCH_SIZE,512,HEIGHT/4,WIDTH/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (BATCH_SIZE,512,HEIGHT/4,WIDTH/4)-> (BATCH_SIZE,512,HEIGHT/2,WIDTH/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (BATCH_SIZE,256,HEIGHT/2,WIDTH/2)-> (BATCH_SIZE,256,HEIGHT,WIDTH)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32,128),
            nn.SiLU(),
            # (BATCH_SIZE,128,HEIGHT,WIDTH)-> (BATCH_SIZE,3,HEIGHT,WIDTH)
            nn.Conv2d(128,3,kernel_size=3,padding=1)
        )
    def forward(self, input:torch.Tensor)->torch.Tensor:
        # x: (batch_size,4,height/8,width/8)
        input/=0.18215
        for module in self:
            input=module(input)
        # (BATCH_SIZE,3,HEIGHT,WIDTH)
        return input
