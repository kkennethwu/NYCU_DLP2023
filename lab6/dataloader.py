import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class iclevrDataset(Dataset):
    def __init__(self, root=None, _transforms=None, mode="train",):
        super().__init__()
        assert mode in ['train', 'test', 'new_test'], "There is no such mode !!!"
        # initialize img_paths
        if mode == 'train':
            with open('train.json', 'r') as json_file:
                self.json_data = json.load(json_file)
            self.img_paths = list(self.json_data.keys())
            self.labels = list(self.json_data.values())
        elif mode == 'test':
            with open('test.json', 'r') as json_file:
                self.json_data = json.load(json_file)
            self.labels = self.json_data
        elif mode == 'new_test':
            with open('new_test.json', 'r') as json_file:
                self.json_data = json.load(json_file)
            self.labels = self.json_data
        # initailize one-hot labels
        self.labels_one_hot = []
        with open('objects.json', 'r') as json_file:
            self.objects_dict = json.load(json_file)
        for label in self.labels:
            label_one_hot = [0] * len(self.objects_dict)
            for text in label:
                label_one_hot[self.objects_dict[text]] = 1
            self.labels_one_hot.append(label_one_hot)
        # initialize others
        self.root = root   
        self.mode = mode
            
    def __len__(self):
        return len(self.labels)      
    
    def __getitem__(self, index):
        if self.mode == 'train':
            transform_img = transforms.Compose([
                transforms.
                transforms.ToTensor(),
                transforms.Resize((128, 128))
            ])
            img_path = os.path.join(self.root, self.img_paths[index])
            img = Image.open(img_path).convert("RGB")
            img = transform_img(img)
            label_one_hot = torch.tensor(np.array(self.labels_one_hot[index]))
            return img, label_one_hot
        elif self.mode == 'test':
            label_one_hot = torch.tensor(np.array(self.labels_one_hot[index]))
            return label_one_hot
        elif self.mode == 'new_test':
            label_one_hot = torch.tensor(np.array(self.labels_one_hot[index]))
            return label_one_hot






