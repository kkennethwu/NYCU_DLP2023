import torch
import torch.nn as nn
from diffusers import UNet2DModel

class Unet(nn.Module):
    def __init__(self, labels_num=24, embedding_label_size=4) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(labels_num, embedding_label_size)
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3+labels_num,
            out_channels=3,
            time_embedding_type="positional",
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    def forward(self, x, t, label):
        bs, c, w, h = x.shape
        # embeded_label = self.label_embedding(labels)
        embeded_label = label.view(bs, label.shape[1], 1, 1).expand(bs, label.shape[1], w, h)
        unet_input = torch.cat((x, embeded_label), 1)
        unet_output = self.model(unet_input, t).sample
        return unet_output
        
        
        

