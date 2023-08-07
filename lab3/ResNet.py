# print("Please define your ResNet in this file.")
import torch

class BasicBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(BasicBlock, self).__init__()
        self.plane = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3,), stride=stride, padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3,), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
 
        self.shortcut = torch.nn.Sequential()
        if input_channel != output_channel:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), stride=stride, bias=False), # when shortcut go through feature maps of two sizes, pserformed with stride of 2
                torch.nn.BatchNorm2d(output_channel) # ?
            )
    
    def forward(self, x):
        x2 = self.shortcut(x)
        x1 = self.plane(x)
        
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
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.layer1 = torch.nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        self.layer2 = torch.nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )
        self.layer3 = torch.nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)    
        )
        self.layer4 = torch.nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)    
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim=1)
        
        return x
        

class Bottleneck(torch.nn.Module):
    def __init__(self, input_channel, output_channel, start_channel, stride):
        super(Bottleneck, self).__init__()
        self.plane = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, start_channel, kernel_size=(1, 1,), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(start_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(start_channel, start_channel, kernel_size=(3, 3,), stride=stride, padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(start_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(start_channel, output_channel, kernel_size=(1, 1,), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(output_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # torch.nn.ReLU(inplace=True)
        )
        self.shortcut = torch.nn.Sequential()
        if input_channel != output_channel:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), stride=stride, bias=False), # when shortcut go through feature maps of two sizes, pserformed with stride of 2
                torch.nn.BatchNorm2d(output_channel, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True) # ?
            )
    
    def forward(self, x):
        x2 = self.shortcut(x)
        x1 = self.plane(x)
        
        x1 += x2
        x1 = torch.nn.functional.relu(x1)
        # print(x1.shape)
        return x1
        
        
class ResNet50(torch.nn.Module):
    def __init__(self, Bottleneck):
        super(ResNet50, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2,), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.layer1 = torch.nn.Sequential(
            Bottleneck(64, 256, start_channel=64, stride=1),
            Bottleneck(256, 256, start_channel=64, stride=1),
            Bottleneck(256, 256, start_channel=64, stride=1)
        )
        self.layer2 = torch.nn.Sequential(
            Bottleneck(256, 512, start_channel=128, stride=2),
            Bottleneck(512, 512, start_channel=128, stride=1),
            Bottleneck(512, 512, start_channel=128, stride=1)
        )
        self.layer3 = torch.nn.Sequential(
            Bottleneck(512, 1024, start_channel=256, stride=2),
            Bottleneck(1024, 1024, start_channel=256, stride=1),
            Bottleneck(1024, 1024, start_channel=256, stride=1),
            Bottleneck(1024, 1024, start_channel=256, stride=1),
            Bottleneck(1024, 1024, start_channel=256, stride=1)
        )
        self.layer4 = torch.nn.Sequential(
            Bottleneck(1024, 2048, start_channel=512, stride=2),
            Bottleneck(2048, 2048, start_channel=512, stride=1),
            Bottleneck(2048, 2048, start_channel=512, stride=1),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # x = torch.nn.functional.softmax(x, dim=1)
        
        return x

class ResNet152(torch.nn.Module):
    def __init__(self, Bottleneck):
        super(ResNet152, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2,), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self.make_layer(Bottleneck, 64, 256, 64, 3, stride=1) 
        self.layer2 = self.make_layer(Bottleneck, 256, 512, 128, 8, stride=2)
        self.layer3 = self.make_layer(Bottleneck, 512, 1024, 256, 36, stride=2)
        self.layer4 = self.make_layer(Bottleneck, 1024, 2048, 512, 3, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
        
        
    def make_layer(self, block, in_channels, out_channel, start_channel, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channel, start_channel, stride=stride))
        for i in range(num_blocks - 1):
            layers.append(block(out_channel, out_channel, start_channel, stride=1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # x = torch.nn.functional.softmax(x, dim=1)
        
        return x      