import torch

class ActivationLayer(torch.nn.Module):
    def __init__(self, activation):
        super(ActivationLayer, self).__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)

class DeepConvNet(torch.nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        
        self.Layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 25, kernel_size=(1, 5)),
            torch.nn.Conv2d(25, 25, kernel_size=(2, 1)),
            torch.nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            ActivationLayer(activation),
            torch.nn.MaxPool2d(kernel_size=(1, 2), padding=0),
            torch.nn.Dropout(p=0.5)
        ).double()
        self.Layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(25, 50, kernel_size=(1, 5)),
            torch.nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            ActivationLayer(activation),
            torch.nn.MaxPool2d(kernel_size=(1, 2), padding=0),
            torch.nn.Dropout(p=0.5)
        ).double()
        self.Layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(50, 100, kernel_size=(1, 5)),
            torch.nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            ActivationLayer(activation),
            torch.nn.MaxPool2d(kernel_size=(1, 2), padding=0),
            torch.nn.Dropout(p=0.5)
        ).double()
        self.Layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(100, 200, kernel_size=(1, 5)),
            torch.nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            ActivationLayer(activation),
            torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            torch.nn.Dropout(p=0.5)
        ).double()
        self.classify = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_features=8600, out_features=2, bias=True),
            # torch.nn.Softmax()
        ).double()
    def forward(self, x):
        x = self.Layer1(x)
        # print(x.shape)
        x = self.Layer2(x)
        # print(x.shape)
        x = self.Layer3(x)
        # print(x.shape)
        x = self.Layer4(x)
        # print(x.shape)
        x = self.classify(x)
        # print(x)
        # x = torch.nn.functional.log_softmax(x)
        # print(x)
        return x
        
        
        
