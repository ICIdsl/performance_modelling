import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features_0 = nn.Conv2d(3, 2, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.features_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.features_3 = nn.Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.features_5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.features_6 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_8 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_10 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_12 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier_1 = nn.Linear(in_features=72, out_features=4096, bias=True)
        self.classifier_4 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.classifier_6 = nn.Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, x):
        x = self.features_0(x)
        x = F.relu(x, inplace=True)
        x = self.features_2(x)
        x = self.features_3(x)
        x = F.relu(x, inplace=True)
        x = self.features_5(x)
        x = self.features_6(x)
        x = F.relu(x, inplace=True)
        x = self.features_8(x)
        x = F.relu(x, inplace=True)
        x = self.features_10(x)
        x = F.relu(x, inplace=True)
        x = self.features_12(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.classifier_4(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.classifier_6(x)
        return x

def alexnet(**kwargs):
    return AlexNet(**kwargs)
