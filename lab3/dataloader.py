import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import random

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    else:
        df = pd.read_csv('resnet_18_test.csv')
        path = df['Path'].tolist()
        return path

class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.mode = mode
        if(self.mode == "train" or self.mode =="valid"):
            self.img_name, self.label = getData(mode)
        else:
            self.img_name = getData(mode)
        
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        img_path = self.root + self.img_name[index]
        img = Image.open(img_path)
        
        angle = random.uniform(0, 180)
        
        if(self.mode == "train"):
            transform = transforms.Compose([
                # transforms.CenterCrop(400),
                transforms.Resize((224, 224)),  # Resize the image to (256, 256) if needed
                # transforms.RandomRotation(angle),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(), # Convert the image to a PyTorch tensor
            ])
            img = transform(img)
            label = self.label[index]
            return img, label            
        elif self.mode == "valid":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize the image to (256, 256) if needed
                transforms.ToTensor(), # Convert the image to a PyTorch tensor
            ])
            img = transform(img)
            label = self.label[index]
            return img, label
        else:
            transform = transforms.Compose([
                transforms.Resize((244, 224)),  # Resize the image to (256, 256) if needed
                transforms.ToTensor(), # Convert the image to a PyTorch tensor
            ])
            img = transform(img)
            return img