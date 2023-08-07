# print("Please define your ResNet in this file.")
import torch

class BasicBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super(BasicBlock, self).__init__()
        self.plane = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3,), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3,), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
 
        self.shortcut = torch.nn.Sequential()
        if input_channel != output_channel:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
                torch.nn.BatchNorm2d(output_channel) # ?
            )
    
    def forward(self, x):
        x1 = self.plane(x)
        x2 = self.shortcut(x)
        x1 += x2
        x1 = torch.nn.functional.relu(x1)
        # print(x1.shape)
        return x1


class ResNet18(torch.nn.Module):
    def __init__(self, BasicBlock):
        super(ResNet18, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2,), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)        
        self.layer1 = torch.nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = torch.nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = torch.nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256)    
        )
        self.layer4 = torch.nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512)    
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x
        
            


        