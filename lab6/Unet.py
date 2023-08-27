import torch
import torch.nn as nn
from diffusers import UNet2DModel

class Unet(nn.Module):
    def __init__(self, labels_num=24, embedding_label_size=10) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(labels_num, embedding_label_size)
        self.model = UNet2DModel(
            in_channels=3,
            out_channels=3,
            time_embedding_type="positional",
        )
    def forward(self, x, t, labels):
        embeded_label = self.label_embedding(labels)
        breakpoint()
        unet_input = torch.cat((x, t, embeded_label))
        breakpoint()
        return x
        
        
        

