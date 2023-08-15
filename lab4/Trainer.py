import os
import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        if (self.kl_anneal_type == "Cyclical") or (self.kl_anneal_type == "Monotonic"):
            self.beta = 0.01
        else: # no kl_annealing
            self.beta = 1
        # raise NotImplementedError

        
    def update(self):
        # TODO
        self.current_epoch += 1
        if self.kl_anneal_type == "Cyclical":
            self.frange_cycle_linear(self.current_epoch, self.beta, stop=1.0, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)
        elif self.kl_anneal_type == "Monotonic":
            self.frange_monotonic_linear(self.current_epoch, self.beta, stop=1.0, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)
        # else 
        # raise NotImplementedError
    
    def get_beta(self):
        # TODO
        return self.beta
        # raise NotImplementedError

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        tmp = 0
        tmp = ((n_iter % n_cycle) / n_cycle) * ratio
        if(tmp <= stop):
            self.beta = tmp
        else:
            self.beta = stop        
        print("\nbeta after update: ", self.beta)
        # raise NotImplementedError
    
    def frange_monotonic_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        tmp = 0
        tmp = (n_iter / n_cycle) * ratio
        if(tmp <= stop):
            self.beta = tmp
        else:
            self.beta = stop

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.current_val_psnr = 0
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        train_loss_list, kl_list, mse_list, epoch_list, tfr_list, beta_list, val_psnr_list = [], [], [], [], [], [], []
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            loss_sum, kl_sum, mse_sum = 0, 0, 0
            train_loader_len = train_loader.__len__()
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, mse_loss, kl_loss = self.training_one_step(img, label, adapt_TeacherForcing)
                loss_sum += loss.item()
                mse_sum += mse_loss.item()
                kl_sum += kl_loss.item()
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            ##### for loss/psnr graph #####
            train_loss_list.append(loss_sum / train_loader_len) 
            mse_list.append(mse_sum / train_loader_len)
            kl_list.append(kl_sum/ train_loader_len)   
            epoch_list.append(i)
            ##### for tfr_beta graph #####
            tfr_list.append(self.tfr)
            beta_list.append(beta)
            
            self.eval()
            ##### for tfr_beta graph
            val_psnr_list.append(self.current_val_psnr)
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        self.plot_loss_curve(train_loss_list, mse_list, kl_list, epoch_list, self.args.kl_anneal_type)
        self.plot_tfr(tfr_list, beta_list, epoch_list)
        self.plot_val_psnr(epoch_list, val_psnr_list)

    def plot_val_psnr(self, epoch_list, val_psnr_list):
        plt.plot(epoch_list, val_psnr_list, label="val_psnr")
        plt.xlabel("epochs")
        plt.title(f"VAL PSNR")
        plt.legend()
        plt.savefig(f"graph/VAL_PSNR")
        plt.close()
    
    def plot_tfr(self, tfr_list, beta_list, epoch_list):
        plt.plot(epoch_list, beta_list, label="beta")
        plt.plot(epoch_list, tfr_list, label="tfr_ratio")
        plt.xlabel("epochs")
        plt.title("TFR ratio and Beta")
        plt.legend()
        plt.savefig("graph/tfr")
        plt.close()
    
    def plot_loss_curve(self, train_loss_list, mse_list, kl_list, epoch_list, kl_anneal_type):
        plt.plot(epoch_list, train_loss_list, label="total_loss")
        plt.plot(epoch_list, mse_list, label="mse_loss")
        plt.plot(epoch_list, kl_list, label="kl_loss")
        plt.xlabel("epochs")
        plt.ylim(0, 1)
        plt.title(f"Loss Curve of {kl_anneal_type}")
        plt.legend()
        plt.savefig(f"graph/loss_curve_{kl_anneal_type}")
        plt.close()
        
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        # img->frame, label->pose
        self.frame_transformation.train()
        self.label_transformation.train()
        self.Gaussian_Predictor.train()
        self.Decoder_Fusion.train()
        self.Generator.train()
        
        mse_loss = 0
        kl_loss = 0
        predicted_next_frame = img[:, 0]
        for i in range(self.train_vi_len - 1):
            current_pose, next_pose = label[:, i], label[:, i+1]
            current_frame, next_frame = predicted_next_frame, img[:, i+1]
            ##### Add Teacher forcing
            if adapt_TeacherForcing: # ?d
                current_frame = img[:, i]  
            ##### KL Loss #####
            encoded_next_frame = self.frame_transformation(next_frame)
            encoded_next_pose = self.label_transformation(next_pose)
            z, mu, logvar = self.Gaussian_Predictor(encoded_next_frame, encoded_next_pose) # z for reparameterization trick
            kl_loss += kl_criterion(mu, logvar, self.batch_size)
            ##### MSE Loss #####
            encoded_current_frame = self.frame_transformation(current_frame)
            decoded_features = self.Decoder_Fusion(encoded_current_frame, encoded_next_pose, z) # param ?
            predicted_next_frame = self.Generator(decoded_features)
            mse_loss += self.mse_criterion(predicted_next_frame, next_frame)
            # breakpoint()
        
        ##### kl_annealing #####
        beta = self.kl_annealing.get_beta()
        loss = mse_loss + beta * kl_loss
        ##### back porpagation #####
        loss.backward()
        self.optimizer_step()
        self.optim.zero_grad()
        return loss, mse_loss, kl_loss
        # raise NotImplementedError
    
    def val_one_step(self, img, label):
        # TODO
        self.frame_transformation.eval()
        self.label_transformation.eval()
        self.Decoder_Fusion.eval()
        self.Generator.eval()
        
        mse_loss = 0
        predicted_next_frame = img[:, 0]
        predicted_img_list = [] # Could add the first frame
        psnr_sum = 0
        index_list = []
        psnr_list = []
        for i in range(self.val_vi_len - 1):
            current_pose, next_pose = label[:, i], label[:, i+1]
            current_frame, next_frame = predicted_next_frame, img[:, i+1]
            
            encoded_current_frame = self.frame_transformation(current_frame)
            encoded_next_pose = self.label_transformation(next_pose)
            # print(encoded_current_frame.shape)
            # print(encoded_next_pose.shape)
            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W) # weired
            z = z.to(self.args.device)
            decoded_features = self.Decoder_Fusion(encoded_current_frame, encoded_next_pose, z) # param ?
            predicted_next_frame = self.Generator(decoded_features)
            mse_loss += self.mse_criterion(predicted_next_frame, next_frame)
            ##### PSNR #####
            psnr_per_frame = Generate_PSNR(predicted_next_frame, next_frame).item()
            psnr_sum += psnr_per_frame
            if self.args.test:
                index_list.append(i)
                psnr_list.append(psnr_per_frame) 
            ##### make gif #####
            if (self.current_epoch == self.args.num_epoch) or self.args.test:
                predicted_img_list.append(predicted_next_frame[0])
        ##### AVG PSNR #####
        self.current_val_psnr = psnr_sum / self.val_vi_len
        print("\nAVG PSNR: ", self.current_val_psnr)
        if self.args.test:
            self.plot_psnr(index_list, psnr_list, round(self.current_val_psnr, 3))
        ##### make gif #####
        if (self.current_epoch == self.args.num_epoch) or self.args.test:
            self.make_gif(predicted_img_list, os.path.join(self.args.save_root, f"epoch={self.current_epoch}_val.gif"))        
        
        return mse_loss
        
        # raise NotImplementedError
    
    def plot_psnr(self, index_list, psnr_list, psnr_avg):
        plt.plot(index_list, psnr_list, label=f"AVG_PSNR: {psnr_avg}")
        plt.xlabel("Frame index")
        plt.ylabel("PSNR")
        plt.title("Per frame Quality (PSNR)")
        plt.legend()
        plt.savefig("graph/per_frame_quality")
        plt.close()
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.tfr_sde:
            tmp_tfr = self.tfr
            tmp_tfr -= 1 / (self.args.num_epoch - self.tfr_sde)
            self.tfr = max(tmp_tfr, 0)
        # print(self.tfr)
        # print(self.tfr_d_step)
        # print(self.tfr_sde)
        
        # raise NotImplementedError
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    # parser.add_argument('--config_file', type=str, help='config file path')
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='NoKL_Annealing',       help="Cyclical, Monotonic, NoKL_Annealing")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    args = parser.parse_args()
    # if args.config_file:
    #     config = configparser.ConfigParser()
    #     config.read(args.config_file)
    #     defaults = {}
    #     defaults.update(dict(config.items("Defaults")))
    #     parser.set_defaults(**defaults)
    #     args = parser.parse_args() # Overwrite arguments
    
    main(args)
