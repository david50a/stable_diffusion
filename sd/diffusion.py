import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention,CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self,n_embd:int)->None:
        super().__init__()
        self.linear_1=nn.Linear(n_embd,4*n_embd)
        self.linear_2=nn.Linear(4*n_embd,4*n_embd)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # (1,320)
        x=self.linear_1(x)
        x=F.silu(x)
        x=self.linear_2(x)
        # (1,1280)
        return x

class SwitchSequential(nn.Sequential):
    def forward(self, x:torch.Tensor,context:torch.Tensor,time:torch.Tensor)->torch.Tensor:
        for layer in self:
            if isinstance(layer,UNET_AttentionsBlock):
                x=layer(x,context)
            elif isinstance(layer,UNET_residualBlock):
                x=layer(x,time)
            else:
                x=layer(x)
        return x

class Upsample(nn.Module):
    def __subclasshook__(self, channels:int):
        super().__init__()
        self.conv=nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # (batch_size, features, height,width) -> (Batch_size, Features, 2*height, 2*width)
        x=F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.encoders=nn.ModuleList([
             # (batch_size),4,height/8,width/8)
             SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
             SwitchSequential(UNET_residualBlock(320,320),UNET_AttentionBlock(8,40)),
             SwitchSequential(UNET_residualBlock(320,320),UNET_AttentionBlock(8,40)),
             # (batch_size),320,height/8,width/8) -> (batch_size),320,height/16,width/16)
             SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
             SwitchSequential(UNET_residualBlock(320,640),UNET_AttentionBlock(8, 80)),
             SwitchSequential(UNET_residualBlock(640, 640),UNET_AttentionBlock(8, 80)),
             # (batch_size),640,height/16,width/16) -> (batch_size),640,height/32,width/32)
             SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
             SwitchSequential(UNET_residualBlock(640, 1280),UNET_AttentionBlock(8, 160)),
             SwitchSequential(UNET_residualBlock(1280, 1280),UNET_AttentionBlock(8, 160)),
             # (batch_size),1280,height/32,width/32) -> (batch_size),1280,height/64,width/64)
             SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
             SwitchSequential(UNET_residualBlock(1280,1280)),
             #  (batch_size),1280,height/64,width/64)-> (batch_size),1280,height/64,width/64)
             SwitchSequential(UNET_residualBlock(1280,1280)),
        ])

        self.bottleneck=SwitchSequential(
            UNET_residualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_residualBlock(1280,1280),
        )
        self.decoder=nn.ModuleList([
            # (batch_size),2560,height/64,width/64)-> (batch_size),1280,height/64,width/64)
            SwitchSequential(UNET_residualBlock(2560,1280)),
            SwitchSequential(UNET_residualBlock(2560,1280)),
            SwitchSequential(UNET_residualBlock(2560,1280),Upsample(1280)),
            SwitchSequential(UNET_residualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_residualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            SwitchSequential(UNET_residualBlock(2560,1280),UNET_AttentionBlock(8,160),Upsample(1280)),
            SwitchSequential(UNET_residualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_residualBlock(1280, 1280), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_residualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNET_residualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 40)),

        ])
        
class Diffusion(nn.Module):
    def __init__(self)->None:
        self.time_embedding=TimeEmbedding(320)
        self.unet=UNET()
        self.final=UNET_OutputLayer(320,4)
    def forward(self,latent:torch.Tensor,context:torch.Tensor,time:torch.Tensor)->torch.Tensor:
        # latent: (batch_size,4,height/8,width/8)
        # context: (batch_size,seq_len,dim)
        # time:(320,4)
        #(1,320)->(1,1200)
        time=self.time_embedding(time)
        # (batch, 4, height/8,width/8) -> (batch, 320, height/8,width/8)
        output=self.unet(latent,context,time)
        # (batch, 320, height/8,width/8) -> (batch, 4, height/8,width/8)
        output=self.final(output)
        return output
