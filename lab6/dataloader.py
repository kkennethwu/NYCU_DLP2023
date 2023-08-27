import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class iclevrDataset(Dataset):
    def __init__(self, root, _transforms=None, mode="train",):
        super().__init__()
        assert mode in ['train', 'test'], "There is no such mode !!!"
        # initialize img_paths
        if mode == 'train':
            with open('train.json', 'r') as json_file:
                self.json_data = json.load(json_file)
            self.img_paths = list(self.json_data.keys())
            self.labels = list(self.json_data.values())
        if mode == 'test':
            with open('test.json', 'r') as json_file:
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
        return len(self.img_paths)      
    
    def __getitem__(self, index):
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            img_path = os.path.join(self.root, self.img_paths[index])
            img = Image.open(img_path)
            img = transform(img)
            label_one_hot = self.labels_one_hot[index]
            return img, label_one_hot
        elif self.mode == 'test':
            label_one_hot = self.labels_one_hot[index]
            print(label_one_hot)
            return label_one_hot
        






