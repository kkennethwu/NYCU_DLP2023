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
from evaluator import evaluation_model
import torchvision.transforms as transforms
import os


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



class ConditionlDDPM():
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        
        self.device = args.device
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_train_timestamps = args.num_train_timestamps
        
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timestamps, beta_schedule="squaredcos_cap_v2")
        self.noise_predicter = Unet(labels_num=24, embedding_label_size=4).to(self.device)
        self.eval_model = evaluation_model()
        
        self.optimizer = torch.optim.Adam(self.noise_predicter.parameters(), lr=self.lr)
        
    def train(self):
        
        train_dataset = iclevrDataset(root="iclevr", mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        
        loss_criterion = nn.MSELoss()
        
        
        # training 
        for epoch in range(1, self.epochs):
            loss_sum = 0
            for x, y in tqdm(train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                noise = torch.randn_like(x)
                timestamp = torch.randint(0, self.num_train_timestamps - 1, (x.shape[0], ), device=self.device).long()
                noise_x = self.noise_scheduler.add_noise(x, noise, timestamp)
                perd_noise = self.noise_predicter(noise_x, timestamp, y)
                
                
                loss = loss_criterion(noise, perd_noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss
            avg_loss = loss_sum / epoch
            print("avg_loss: ", avg_loss.item())
            self.writer.add_scalar("avg_loss", avg_loss, epoch)
            # validation
            if(epoch % 10 == 0):
                eval_acc = self.evaluate(epoch, test_what="test")
                self.writer.add_scalar("eval_accuracy", eval_acc, epoch)
                # self.save(os.path.join(self.args.save_root, f"epoch={epoch}.ckpt"))

    def evaluate(self, epoch="final", test_what="test"):
        test_dataset = iclevrDataset(mode=f"{test_what}")
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        for y in test_dataloader:
            y = y.to(self.device)
            x = torch.randn(32, 3, 64, 64).to(self.device)
            for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                with torch.no_grad():
                    pred_noise = self.noise_predicter(x, t, y)
                x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
            # compute accuracy using pre-trained model
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            acc = self.eval_model.eval(images=x.detach(), labels=y)
            print(f"accuracy of {test_what}.json on epoch {epoch}: ", round(acc, 3))
            generated_grid_imgs = make_grid(x.detach())
            save_image(generated_grid_imgs, f"eval/test_{epoch}.jpg")
        return round(acc, 3)
    
    # def load_checkpoint(self):
    #     if self.ckpt_path != None:
    #         checkpoint = torch.load(self.args.ckpt_path)
    #         self.load_state_dict(checkpoint['state_dict'], strict=True) 
    #         self.lr = checkpoint['lr']
            
    #         self.optimizer      = torch.optim.Adam(self.noise_predicter.parameters(), lr=self.args.lr)
    #         # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
    #         # self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
    #         self.epoch = checkpoint['last_epoch']
    
    # def save(self, path):
    #     torch.save({
    #         "state_dict": self.state_dict(),
    #         "optimizer": self.state_dict(),  
    #         "lr"        : self.scheduler.get_last_lr()[0],
    #         "tfr"       :   self.tfr,
    #         "last_epoch": self.current_epoch
    #     }, path)
    #     print(f"save ckpt to {path}")
   
   
            
def main(args):
    writer = SummaryWriter()
    conditionlDDPM = ConditionlDDPM(args, writer)
    conditionlDDPM.train()
    conditionlDDPM.evaluate(epoch=150, test_what="new_test")
    writer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["cuda:0", "cuda:1", "cpu"], default="cuda:0")
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_train_timestamps', type=int, default=1000)
    # ckpt
    # parser.add_argument('--ckpt_path', type=str, default=None)
    # parser.add_argument('--save_root', type=str, default="ckpt")
    # tensorboard args
    # parser.add_argument('--log_dir', type=str, default="logs")
    
    args = parser.parse_args()
    
    main(args)



