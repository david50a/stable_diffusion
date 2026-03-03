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
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            # Check if the layer is a Residual Block (needs time)
            if isinstance(layer, UNET_residualBlock):
                x = layer(x, time)
            # Check if the layer is an Attention Block (needs context)
            elif isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            # Standard layers (Conv2d, Upsample) only take x
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv=nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # (batch_size, features, height,width) -> (Batch_size, Features, 2*height, 2*width)
        x=F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNET_residualBlock(nn.Module):
    # Change n_time to 1280
    def __init__(self, in_channels: int, out_channels: int, n_time=1280) -> None:
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        # Change this from nn.Linear to nn.Conv2d
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, feature, time) -> torch.Tensor:
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        # Add time embedding
        time = F.silu(time)
        time = self.linear_time(time)

        # Merge feature and time
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)  # Using the corrected conv layer

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, d_head: int, d_context=768) -> None:
        super().__init__()
        # Total channels = number of heads * dimension per head
        channels = n_head * d_head

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)  # Corrected name from linearnorm
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)
        # GEGLU uses 8 * channels because it chunks into 4 * channels
        self.linear_geglu_1 = nn.Linear(channels, 8 * channels)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x:torch.Tensor,context)->torch.Tensor:
        # x: (batch_size, in_channels,height,width)
        # context: (batch_size,seq_len,dim)
        residue_long=x
        x=self.groupnorm(x)
        x=self.conv_input(x)
        n,c,h,w=x.shape
        # (batch_size, features,height,width) -> (batch_size, features,height*width)
        x=x.view((n,c,h*w))
        # (batch_size, features,height,width) -> (batch_size,height*width,features)
        x=x.transpose(-1,-2)
        # Normalization + self attention with skip connection
        residue_short=x
        x=self.layernorm_1(x)
        self.attention_1(x)
        x+=residue_short
        # Normalization + cross attention with skip connection
        self.layernorm_2(x)
        # cross attention
        self.attention_2(x,context)
        x+=residue_short
        residue_short=x
        # Normalization + FF with DeGLU and skip connection
        x=self.layernorm_3(x)
        x,gate=self.linear_geglu_1(x).chunk(2, dim=-1)
        x=x*F.gelu(gate)
        x=self.linear_geglu_2(x)
        x+=residue_short
        #(batch_size,height*width,feature) -> (batch_size,feature,height,width)
        x=x.transpose(-1,-2)
        x=x.view((n,c,h,w))
        return self.conv_output(x)+residue_long

class UNET_OutputLayer(nn.Module):
    def __init__(self,in_channels:int,out_channel:int)->None:
        super().__init__()
        self.groupnorm=nn.GroupNorm(32,in_channels)
        self.conv=nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
         # x: (batch_size, 320, height/8, width/8)
        x=self.groupnorm(x)
        x=F.silu(x)
        x=self.conv(x)
         #(batch_size,4,height/8,width/8)
        return x

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
        self.decoders = nn.ModuleList([
            # (batch, 1280 + 1280, 8, 8) -> (batch, 1280, 8, 8)
            SwitchSequential(UNET_residualBlock(2560, 1280)),
            SwitchSequential(UNET_residualBlock(2560, 1280)),
            SwitchSequential(UNET_residualBlock(2560, 1280), Upsample(1280)),

            # (batch, 1280 + 1280, 16, 16) -> (batch, 1280, 16, 16)
            SwitchSequential(UNET_residualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_residualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            # This block is the one causing the "1920" vs "2560" error (Index 5)
            SwitchSequential(UNET_residualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            # (batch, 640 + 1280, 32, 32) -> (batch, 640, 32, 32)
            SwitchSequential(UNET_residualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            # This block is the "640" vs "1280" error (Index 7)
            SwitchSequential(UNET_residualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_residualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            # (batch, 320 + 640, 64, 64) -> (batch, 320, 64, 64)
            SwitchSequential(UNET_residualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            # This is the "320" vs "640" mismatch in Attention (Index 10)
            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_residualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        skip_connections = []
        for i, layers in enumerate(self.encoders):
            x = layers(x, context, time)
            # Save skip connections for ALL layers
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Now the stack will only contain the 9 layers that match
            # your 12 decoder entries minus the Upsamples.
            if len(skip_connections) > 0:
                x = torch.cat([x, skip_connections.pop()], dim=1)
            x = layers(x, context, time)

        return x

class Diffusion(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.time_embedding=TimeEmbedding(320)
        self.unet=UNET()
        self.final=UNET_OutputLayer(320,4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # 1. Transform time (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # 2. Pass to UNET (where the actual encoder/decoder loops live)
        output = self.unet(latent, context, time)

        # 3. Final projection to 4 channels
        return self.final(output)
