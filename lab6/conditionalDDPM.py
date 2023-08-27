from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from matplotlib import pyplot as plt
import torch
import numpy as np
from dataloader import iclevrDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torch.nn as nn
from tqdm.auto import tqdm
import argparse
from Unet import Unet
from torch.utils.tensorboard import SummaryWriter


# scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
# model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
# scheduler.set_timesteps(50)

# sample_size = model.config.sample_size
# noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
# input = noise

# for t in scheduler.timesteps:
#     with torch.no_grad():
#         noisy_residual = model(input, t).sample
#         prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
#         input = prev_noisy_sample

# image = (input / 2 + 0.5).clamp(0, 1)
# image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
# image = Image.fromarray((image * 255).round().astype("uint8"))
# image.save("test.jpg")





def train(args, writer):
    epochs = args.epochs
    device = args.device
    lr = args.lr
    num_train_timestamp = args.num_train_timestamps
    
    train_dataset = iclevrDataset(root="iclevr", mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timestamps, beta_schedule="squaredcos_cap_v2")
    noise_predicter = Unet(labels_num=24, embedding_label_size=10).to(device)
    
    loss_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(noise_predicter.parameters(), lr=lr)
    
    for epoch in range(1, epochs):
        loss_sum = 0
        for x, y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            noise = torch.randn_like(x)
            timestamp = torch.randint(0, num_train_timestamp - 1, (x.shape[0], ), device=device).long()
            noise_x = noise_scheduler.add_noise(x, noise, timestamp)
            perd_noise = noise_predicter(noise_x, timestamp, y)
            
            
            loss = loss_criterion(noise, perd_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        avg_loss = loss_sum / epoch
        print("avg_loss: ", avg_loss)
        writer.add_scalar("avg_loss", avg_loss)
    writer.close()         
            
def main(args):
    writer = SummaryWriter()
    train(args, writer)      
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["cuda:0", "cuda:1", "cpu"], default="cuda")
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_train_timestamps', type=int, default=1000)
    # tensorboard args
    parser.add_argument('--log_dir', type=str, default="logs")
    
    args = parser.parse_args()
    
    main(args)



