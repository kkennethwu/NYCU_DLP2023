import torch

class DeppConvNet(torch.nn.Module):
    def __init__(self):
        
        super(DeppConvNet, self).__init__()
        
        self.Layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 25, kernel_size=(1, 5)),
            torch.nn.Conv2d(25, 25, kernel_size=(2, 1)),
            torch.nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            torch.nn.ELU(alpha=1.0),
            torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            torch.nn.Dropout(p=0.25)
        ).double()
        
    def forward(self, x):
        x = self.Layer1(x)
        return x
        
        
        
