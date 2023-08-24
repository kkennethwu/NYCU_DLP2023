from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scheduler = DDPMScheduler()
model = UNet2DModel()





