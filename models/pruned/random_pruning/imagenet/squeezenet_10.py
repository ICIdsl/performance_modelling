import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features_0 = nn.Conv2d(3, 88, kernel_size=(7, 7), stride=(2, 2))
        self.features_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.features_3_squeeze = nn.Conv2d(88, 14, kernel_size=(1, 1), stride=(1, 1))
        self.features_3_expand1x1 = nn.Conv2d(14, 60, kernel_size=(1, 1), stride=(1, 1))
        self.features_3_expand3x3 = nn.Conv2d(14, 59, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_4_squeeze = nn.Conv2d(119, 15, kernel_size=(1, 1), stride=(1, 1))
        self.features_4_expand1x1 = nn.Conv2d(15, 59, kernel_size=(1, 1), stride=(1, 1))
        self.features_4_expand3x3 = nn.Conv2d(15, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_5_squeeze = nn.Conv2d(120, 31, kernel_size=(1, 1), stride=(1, 1))
        self.features_5_expand1x1 = nn.Conv2d(31, 118, kernel_size=(1, 1), stride=(1, 1))
        self.features_5_expand3x3 = nn.Conv2d(31, 125, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.features_7_squeeze = nn.Conv2d(243, 30, kernel_size=(1, 1), stride=(1, 1))
        self.features_7_expand1x1 = nn.Conv2d(30, 123, kernel_size=(1, 1), stride=(1, 1))
        self.features_7_expand3x3 = nn.Conv2d(30, 115, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_8_squeeze = nn.Conv2d(238, 46, kernel_size=(1, 1), stride=(1, 1))
        self.features_8_expand1x1 = nn.Conv2d(46, 177, kernel_size=(1, 1), stride=(1, 1))
        self.features_8_expand3x3 = nn.Conv2d(46, 180, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_9_squeeze = nn.Conv2d(357, 45, kernel_size=(1, 1), stride=(1, 1))
        self.features_9_expand1x1 = nn.Conv2d(45, 178, kernel_size=(1, 1), stride=(1, 1))
        self.features_9_expand3x3 = nn.Conv2d(45, 177, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_10_squeeze = nn.Conv2d(355, 60, kernel_size=(1, 1), stride=(1, 1))
        self.features_10_expand1x1 = nn.Conv2d(60, 235, kernel_size=(1, 1), stride=(1, 1))
        self.features_10_expand3x3 = nn.Conv2d(60, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_11 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.features_12_squeeze = nn.Conv2d(475, 62, kernel_size=(1, 1), stride=(1, 1))
        self.features_12_expand1x1 = nn.Conv2d(62, 235, kernel_size=(1, 1), stride=(1, 1))
        self.features_12_expand3x3 = nn.Conv2d(62, 236, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.classifier_1 = nn.Conv2d(471, 1000, kernel_size=(1, 1), stride=(1, 1))
        self.classifier_3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.features_0(x)
        x = F.relu(x, inplace=True)
        x = self.features_2(x)
        x = self.features_3_squeeze(x)
        x = F.relu(x, inplace=True)
        x_0 = x
        x_0 = self.features_3_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_3_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x, inplace=True)
        x = self.features_4_squeeze(x)
        x = F.relu(x, inplace=True)
        x_0 = x
        x_0 = self.features_4_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_4_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x, inplace=True)
        x = self.features_5_squeeze(x)
        x = F.relu(x, inplace=True)
        x_0 = x
        x_0 = self.features_5_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_5_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x, inplace=True)
        x = self.features_6(x)
        x = self.features_7_squeeze(x)
        x = F.relu(x, inplace=True)
        x_0 = x
        x_0 = self.features_7_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_7_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x, inplace=True)
        x = self.features_8_squeeze(x)
        x = F.relu(x, inplace=True)
        x_0 = x
        x_0 = self.features_8_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_8_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x, inplace=True)
        x = self.features_9_squeeze(x)
        x = F.relu(x, inplace=True)
        x_0 = x
        x_0 = self.features_9_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_9_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x, inplace=True)
        x = self.features_10_squeeze(x)
        x = F.relu(x, inplace=True)
        x_0 = x
        x_0 = self.features_10_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_10_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x, inplace=True)
        x = self.features_11(x)
        x = self.features_12_squeeze(x)
        x = F.relu(x, inplace=True)
        x_0 = x
        x_0 = self.features_12_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_12_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x, inplace=True)
        x = self.classifier_1(x)
        x = F.relu(x, inplace=True)
        x = self.classifier_3(x)
        return x.squeeze(2).squeeze(2)

def squeezenet(**kwargs):
    return SqueezeNet(**kwargs)
