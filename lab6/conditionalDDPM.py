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
from diffusers.optimization import get_cosine_schedule_with_warmup


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
        self.svae_root = args.save_root
        self.label_embeding_size = args.label_embeding_size
        
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timestamps, beta_schedule="squaredcos_cap_v2")
        self.noise_predicter = Unet(labels_num=24, embedding_label_size=self.label_embeding_size).to(self.device)
        self.eval_model = evaluation_model()
        
        self.train_dataset = iclevrDataset(root="iclevr", mode="train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # self.optimizer = torch.optim.Adam(self.noise_predicter.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.noise_predicter.parameters(), lr=self.lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=len(self.train_dataloader) * self.epochs,
            num_cycles=50
        )
        
    def train(self):
        loss_criterion = nn.MSELoss()
        # training 
        for epoch in range(1, self.epochs+1):
            loss_sum = 0
            print(f"#################### epoch: {epoch}, lr {self.lr} ####################")
            for x, y in tqdm(self.train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                noise = torch.randn_like(x)
                timestamp = torch.randint(0, self.num_train_timestamps - 1, (x.shape[0], ), device=self.device).long()
                noise_x = self.noise_scheduler.add_noise(x, noise, timestamp)
                perd_noise = self.noise_predicter(noise_x, timestamp, y)
                
                
                loss = loss_criterion(perd_noise, noise)
                loss.backward()
                nn.utils.clip_grad_value_(self.noise_predicter.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.lr = self.lr_scheduler.get_last_lr()[0]
                loss_sum += loss.item()
            avg_loss = loss_sum / len(self.train_dataloader)
            print("avg_loss: ", avg_loss)
            self.writer.add_scalar("avg_loss", avg_loss, epoch)
            # validation
            if(epoch % 5 == 0 or epoch == 1):
                eval_acc = self.evaluate(epoch, test_what="test")
                self.writer.add_scalar("test_accuracy", eval_acc, epoch)
                eval_acc = self.evaluate(epoch, test_what="new_test")
                self.writer.add_scalar("new_test_accuracy", eval_acc, epoch)
            if (epoch == 1 or epoch % 10 == 0):
                self.save(os.path.join(self.args.ckpt_root, f"epoch={epoch}.ckpt"), epoch)
                print("save ckpt")
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
            # trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # trans_x = trans(x.detach())
            acc = self.eval_model.eval(images=x.detach(), labels=y)
            denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
            print(f"accuracy of {test_what}.json on epoch {epoch}: ", round(acc, 3))
            generated_grid_imgs = make_grid(denormalized_x)
            save_image(generated_grid_imgs, f"{self.svae_root}/{test_what}_{epoch}.jpg")
        return round(acc, 3)
    
    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.noise_predicter = checkpoint["noise_predicter"]
            self.noise_scheduler = checkpoint["noise_scheduler"]
            self.optimizer = checkpoint["optimizer"]
            self.lr = checkpoint["lr"]
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            # self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.epoch = checkpoint['last_epoch']
    
    def save(self, path, epoch):
        torch.save({
            "noise_predicter": self.noise_predicter,
            "noise_scheduler": self.noise_scheduler,
            "optimizer": self.optimizer,
            "lr"        : self.lr,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "last_epoch": epoch
        }, path)
        print(f"save ckpt to {path}")
    def progressive_generate_image(self):
        label_one_hot = [0] * 24
        label_one_hot[2] = 1
        label_one_hot[19] = 1
        label_one_hot[3] = 1
        label_one_hot = torch.tensor(label_one_hot).to( self.device)
        label_one_hot = torch.unsqueeze(label_one_hot, 0)
        # breakpoint()
        x = torch.randn(1, 3, 64, 64).to(self.device)
        img_list = []
        for i, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
            with torch.no_grad():
                pred_noise = self.noise_predicter(x, t, label_one_hot)
            x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
            if(t % 100 == 0):
                denormalized_x = (x.detach() / 2 + 0.5).clamp(0, 1)
                save_image(denormalized_x, f"{self.args.save_root}/{t}.jpg")
                img_list.append(denormalized_x)
        grid_img = make_grid(torch.cat(img_list, dim=0), nrow=5)
        save_image(grid_img, f"{self.args.save_root}/progressive_genrate_image.jpg")
            
def main(args):
    writer = SummaryWriter()
    conditionlDDPM = ConditionlDDPM(args, writer)
    if args.test_only:
        conditionlDDPM.load_checkpoint()
        conditionlDDPM.progressive_generate_image()
    else:
        conditionlDDPM.train()
    # conditionlDDPM.evaluate(epoch=150, test_what="new_test")
    writer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["cuda:0", "cuda:1", "cpu", "cuda"], default="cuda")
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_train_timestamps', type=int, default=1000)
    parser.add_argument('--lr_warmup_steps', default=0, type=int)
    parser.add_argument('--save_root', type=str, default="eval")
    parser.add_argument('--label_embeding_size', type=int, default=4)
    # ckpt
    parser.add_argument('--ckpt_root', type=str, default="ckpt") # fro save
    parser.add_argument('--ckpt_path', type=str, default=None) # for load
    # parser.add_argument('--save_root', type=str, default="ckpt")
    # tensorboard args
    # parser.add_argument('--log_dir', type=str, default="logs")
    
    args = parser.parse_args()
    
    main(args)



