# U-Net model for computer vision 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groupnorm_num_group):
        super().__init__()
        # groupnorm is used to normalize the input
        self.groupnorm1 = nn.GroupNorm(groupnorm_num_group, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.groupnorm2 = nn.GroupNorm(groupnorm_num_group, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        # Residual connection
        residual = x
        x = F.silu(self.groupnorm1(x))
        x = self.conv1(x)
        x = F.silu(self.groupnorm2(x))
        x = self.conv2(x)
        x = x + self.residual_conv(residual)
        return x
# upsampling block
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate= False):
        super().__init__()
        if interpolate:
            self.Upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
            )
        else:
            self.Upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.Upsample(x)
        return x
# U-Net class
class UNet(nn.Module):
    def __init__(self, 
                 in_channels = 3, 
                 num_classes = 150,
                 start_dim = 64,
                 dim_mults = [1,2,4,8], 
                 residual_block_per_group = 1,
                 groupnorm_num_group = 16,
                 interpolate_upsample=False):
        super().__init__()
        self.input_image_channels = in_channels
        self.interpolate = interpolate_upsample
        channel_sizes = [start_dim * i for i in dim_mults]
        starting_channel_size, ending_channel_size = channel_sizes[0], channel_sizes[-1]
        # encoder config
        self.encoder_config = []

        for idx, d in enumerate(channel_sizes):
            for _ in range(residual_block_per_group):
                self.encoder_config.append(((d,d), "residual"))
            self.encoder_config.append(((d, d), "downsample"))
            if idx < len(channel_sizes) - 1:
                self.encoder_config.append(((d, channel_sizes[idx+1]), "residual"))    
        #print(f"encoder_config: {self.encoder_config}")
        # Bottleneck Config
        self.bottleneck_config = []
        for _ in range(residual_block_per_group):
            self.bottleneck_config.append(((ending_channel_size, ending_channel_size), "residual"))
        #print(f"bottleneck_cofig: {self.bottleneck_config}")
        out_dim = ending_channel_size
        reversed_encoder_config = self.encoder_config[::-1]
        #print(f"reversed_encoder_config: {reversed_encoder_config}")
        # decoder config
        self.decoder_config = []
        for idx, (config, block_type) in enumerate(reversed_encoder_config):
            #print(f"Congig: {config}, Block: {block_type}")
            enc_in_channels, enc_out_channels = config
            concat_num_channels = out_dim + enc_out_channels 
            if block_type == "residual":
                self.decoder_config.append(((concat_num_channels, enc_in_channels), "residual"))
            elif block_type == "downsample":
                self.decoder_config.append(((enc_in_channels,enc_in_channels), "upsample"))
            out_dim = enc_in_channels
        concat_num_channels = starting_channel_size*2
        self.decoder_config.append(((concat_num_channels, starting_channel_size), "residual"))
        #print(f"decoder_config: {self.decoder_config}")

        ### ACTUAL MODEL ###
        self.conv_in_proj = nn.Conv2d(self.input_image_channels, starting_channel_size, kernel_size=3, padding="same")
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for config, block_type in self.encoder_config:
            in_channels, out_channels = config
            if block_type == "residual":
                self.encoder.append(ResidualBlock(in_channels, out_channels, groupnorm_num_group))
            elif block_type == "downsample":
                self.encoder.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1))
                #self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for config, block_type in self.bottleneck_config:
            in_channels, out_channels = config
            if block_type == "residual":
                self.bottleneck.append(ResidualBlock(in_channels, out_channels, groupnorm_num_group))
                #self.bottleneck.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same"))
        for config, block_type in self.decoder_config:
            in_channels, out_channels = config
            if block_type == "residual":
                self.decoder.append(ResidualBlock(in_channels, out_channels, groupnorm_num_group))
            elif block_type == "upsample":
                self.decoder.append(UpsampleBlock(in_channels, out_channels, self.interpolate))
                #self.decoder.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2))
        self.conv_out_proj = nn.Conv2d(starting_channel_size, num_classes, kernel_size=1)
        #self.sigmoid = nn.Sigmoid()
        #summary(self, (3, 64, 64), device="cpu")
        #print(f"Encoder: {self.encoder}")
        #print(f"Bottleneck: {self.bottleneck}")
        #print(f"Decoder: {self.decoder}")
    def forward(self, x):
        residuals = []
        x = self.conv_in_proj(x)
        residuals.append(x)

        # Encoder - save features before downsampling
        for i, block in enumerate(self.encoder):
            if isinstance(block, ResidualBlock):
                x = block(x)
                residuals.append(x)
            else:  # Downsample block
                x = block(x)
        
        # Bottleneck 
        for block in self.bottleneck:
            x = block(x)
        
        # Decoder - use skip connections properly
        for block in self.decoder:
            if isinstance(block, UpsampleBlock):
                x = block(x)
            elif isinstance(block, ResidualBlock):
                if residuals:
                    skip = residuals.pop()
                    #print(f"skip shape: {skip.shape}, shape of x: {x.shape}")
                    # Ensure spatial dimensions match
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                    x = torch.cat([x, skip], dim=1)
                x = block(x)
        
        # Output projection
        x = self.conv_out_proj(x)
        return x

if __name__ == "__main__":  
    # create a dummy input
    x = torch.randn(4, 3, 128, 128).to(device)
    # create a model
    model = UNet().to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    summary(model, (3, 128, 128), device="cpu")    

