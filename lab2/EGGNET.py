import torch

class EGGNet(torch.nn.Module):
    def __init__(self):
        super(EGGNet, self).__init__()
        
        self.firstconv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            torch.nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ELU(alpha=1.0),
            torch.nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            torch.nn.Dropout(p=0.25)
        )
        self.seperableConv = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ELU(alpha=1.0),
            torch.nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            torch.nn.Dropout(p=0.25)
        )
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(in_features=736, out_features=1, bias=True)
        )
        
    def forward(self, x):
        # print(x.shape)
        x = self.firstconv(x)
        # print(x.shape)
        x = self.depthwiseConv(x)
        # print(x.shape)
        x = self.seperableConv(x)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.classify(x)
        
        return x
        
        
    