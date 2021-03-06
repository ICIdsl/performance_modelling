import torch
import torch.nn as nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1_conv = nn.Conv2d(3, 55, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1_bn = nn.BatchNorm2d(55, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_conv = nn.Conv2d(55, 53, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2_bn = nn.BatchNorm2d(53, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_conv = nn.Conv2d(53, 175, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_bn = nn.BatchNorm2d(175, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception3a_branch1_conv = nn.Conv2d(175, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch1_bn = nn.BatchNorm2d(56, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch2_0_conv = nn.Conv2d(175, 82, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch2_0_bn = nn.BatchNorm2d(82, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch2_1_conv = nn.Conv2d(82, 115, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3a_branch2_1_bn = nn.BatchNorm2d(115, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch3_0_conv = nn.Conv2d(175, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch3_0_bn = nn.BatchNorm2d(15, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch3_1_conv = nn.Conv2d(15, 28, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3a_branch3_1_bn = nn.BatchNorm2d(28, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception3a_branch4_1_conv = nn.Conv2d(175, 29, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch4_1_bn = nn.BatchNorm2d(29, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch1_conv = nn.Conv2d(228, 111, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch1_bn = nn.BatchNorm2d(111, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch2_0_conv = nn.Conv2d(228, 111, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch2_0_bn = nn.BatchNorm2d(111, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch2_1_conv = nn.Conv2d(111, 169, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3b_branch2_1_bn = nn.BatchNorm2d(169, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch3_0_conv = nn.Conv2d(228, 31, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch3_0_bn = nn.BatchNorm2d(31, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch3_1_conv = nn.Conv2d(31, 81, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3b_branch3_1_bn = nn.BatchNorm2d(81, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception3b_branch4_1_conv = nn.Conv2d(228, 54, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch4_1_bn = nn.BatchNorm2d(54, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception4a_branch1_conv = nn.Conv2d(415, 172, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch1_bn = nn.BatchNorm2d(172, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch2_0_conv = nn.Conv2d(415, 81, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch2_0_bn = nn.BatchNorm2d(81, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch2_1_conv = nn.Conv2d(81, 182, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4a_branch2_1_bn = nn.BatchNorm2d(182, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch3_0_conv = nn.Conv2d(415, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch3_0_bn = nn.BatchNorm2d(15, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch3_1_conv = nn.Conv2d(15, 44, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4a_branch3_1_bn = nn.BatchNorm2d(44, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4a_branch4_1_conv = nn.Conv2d(415, 57, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch4_1_bn = nn.BatchNorm2d(57, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch1_conv = nn.Conv2d(455, 140, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch1_bn = nn.BatchNorm2d(140, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch2_0_conv = nn.Conv2d(455, 100, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch2_0_bn = nn.BatchNorm2d(100, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch2_1_conv = nn.Conv2d(100, 205, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4b_branch2_1_bn = nn.BatchNorm2d(205, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch3_0_conv = nn.Conv2d(455, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch3_0_bn = nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch3_1_conv = nn.Conv2d(24, 57, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4b_branch3_1_bn = nn.BatchNorm2d(57, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4b_branch4_1_conv = nn.Conv2d(455, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch4_1_bn = nn.BatchNorm2d(56, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch1_conv = nn.Conv2d(458, 114, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch1_bn = nn.BatchNorm2d(114, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch2_0_conv = nn.Conv2d(458, 111, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch2_0_bn = nn.BatchNorm2d(111, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch2_1_conv = nn.Conv2d(111, 231, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4c_branch2_1_bn = nn.BatchNorm2d(231, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch3_0_conv = nn.Conv2d(458, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch3_0_bn = nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch3_1_conv = nn.Conv2d(24, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4c_branch3_1_bn = nn.BatchNorm2d(61, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4c_branch4_1_conv = nn.Conv2d(458, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch4_1_bn = nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch1_conv = nn.Conv2d(466, 102, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch1_bn = nn.BatchNorm2d(102, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch2_0_conv = nn.Conv2d(466, 131, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch2_0_bn = nn.BatchNorm2d(131, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch2_1_conv = nn.Conv2d(131, 251, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4d_branch2_1_bn = nn.BatchNorm2d(251, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch3_0_conv = nn.Conv2d(466, 29, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch3_0_bn = nn.BatchNorm2d(29, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch3_1_conv = nn.Conv2d(29, 54, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4d_branch3_1_bn = nn.BatchNorm2d(54, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4d_branch4_1_conv = nn.Conv2d(466, 57, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch4_1_bn = nn.BatchNorm2d(57, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch1_conv = nn.Conv2d(464, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch1_bn = nn.BatchNorm2d(232, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch2_0_conv = nn.Conv2d(464, 140, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch2_0_bn = nn.BatchNorm2d(140, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch2_1_conv = nn.Conv2d(140, 282, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4e_branch2_1_bn = nn.BatchNorm2d(282, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch3_0_conv = nn.Conv2d(464, 28, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch3_0_bn = nn.BatchNorm2d(28, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch3_1_conv = nn.Conv2d(28, 110, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4e_branch3_1_bn = nn.BatchNorm2d(110, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4e_branch4_1_conv = nn.Conv2d(464, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch4_1_bn = nn.BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception5a_branch1_conv = nn.Conv2d(736, 229, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch1_bn = nn.BatchNorm2d(229, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch2_0_conv = nn.Conv2d(736, 143, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch2_0_bn = nn.BatchNorm2d(143, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch2_1_conv = nn.Conv2d(143, 284, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5a_branch2_1_bn = nn.BatchNorm2d(284, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch3_0_conv = nn.Conv2d(736, 29, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch3_0_bn = nn.BatchNorm2d(29, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch3_1_conv = nn.Conv2d(29, 117, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5a_branch3_1_bn = nn.BatchNorm2d(117, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception5a_branch4_1_conv = nn.Conv2d(736, 113, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch4_1_bn = nn.BatchNorm2d(113, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch1_conv = nn.Conv2d(743, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch1_bn = nn.BatchNorm2d(336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch2_0_conv = nn.Conv2d(743, 167, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch2_0_bn = nn.BatchNorm2d(167, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch2_1_conv = nn.Conv2d(167, 344, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5b_branch2_1_bn = nn.BatchNorm2d(344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch3_0_conv = nn.Conv2d(743, 41, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch3_0_bn = nn.BatchNorm2d(41, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch3_1_conv = nn.Conv2d(41, 109, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5b_branch3_1_bn = nn.BatchNorm2d(109, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception5b_branch4_1_conv = nn.Conv2d(743, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch4_1_bn = nn.BatchNorm2d(116, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=905, out_features=1000, bias=True)

    def transform_input(self, x):
        x_ch0 = torch.unsqueeze(x[:,0],1) * (0.229/0.5) + (0.485-0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:,1],1) * (0.224/0.5) + (0.456-0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:,2],1) * (0.225/0.5) + (0.406-0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self.transform_input(x)
        x = self.conv1_conv(x)
        x = self.conv1_bn(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool1(x)
        x = self.conv2_conv(x)
        x = self.conv2_bn(x)
        x = F.relu(x, inplace=True)
        x = self.conv3_conv(x)
        x = self.conv3_bn(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool2(x)
        x_0 = x
        x_0 = self.inception3a_branch1_conv(x_0)
        x_0 = self.inception3a_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception3a_branch2_0_conv(x_1)
        x_1 = self.inception3a_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception3a_branch2_1_conv(x_1)
        x_1 = self.inception3a_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception3a_branch3_0_conv(x_2)
        x_2 = self.inception3a_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception3a_branch3_1_conv(x_2)
        x_2 = self.inception3a_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception3a_branch4_0(x_3)
        x_3 = self.inception3a_branch4_1_conv(x_3)
        x_3 = self.inception3a_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x_0 = x
        x_0 = self.inception3b_branch1_conv(x_0)
        x_0 = self.inception3b_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception3b_branch2_0_conv(x_1)
        x_1 = self.inception3b_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception3b_branch2_1_conv(x_1)
        x_1 = self.inception3b_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception3b_branch3_0_conv(x_2)
        x_2 = self.inception3b_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception3b_branch3_1_conv(x_2)
        x_2 = self.inception3b_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception3b_branch4_0(x_3)
        x_3 = self.inception3b_branch4_1_conv(x_3)
        x_3 = self.inception3b_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x = self.maxpool3(x)
        x_0 = x
        x_0 = self.inception4a_branch1_conv(x_0)
        x_0 = self.inception4a_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception4a_branch2_0_conv(x_1)
        x_1 = self.inception4a_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception4a_branch2_1_conv(x_1)
        x_1 = self.inception4a_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception4a_branch3_0_conv(x_2)
        x_2 = self.inception4a_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception4a_branch3_1_conv(x_2)
        x_2 = self.inception4a_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception4a_branch4_0(x_3)
        x_3 = self.inception4a_branch4_1_conv(x_3)
        x_3 = self.inception4a_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x_0 = x
        x_0 = self.inception4b_branch1_conv(x_0)
        x_0 = self.inception4b_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception4b_branch2_0_conv(x_1)
        x_1 = self.inception4b_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception4b_branch2_1_conv(x_1)
        x_1 = self.inception4b_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception4b_branch3_0_conv(x_2)
        x_2 = self.inception4b_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception4b_branch3_1_conv(x_2)
        x_2 = self.inception4b_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception4b_branch4_0(x_3)
        x_3 = self.inception4b_branch4_1_conv(x_3)
        x_3 = self.inception4b_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x_0 = x
        x_0 = self.inception4c_branch1_conv(x_0)
        x_0 = self.inception4c_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception4c_branch2_0_conv(x_1)
        x_1 = self.inception4c_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception4c_branch2_1_conv(x_1)
        x_1 = self.inception4c_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception4c_branch3_0_conv(x_2)
        x_2 = self.inception4c_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception4c_branch3_1_conv(x_2)
        x_2 = self.inception4c_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception4c_branch4_0(x_3)
        x_3 = self.inception4c_branch4_1_conv(x_3)
        x_3 = self.inception4c_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x_0 = x
        x_0 = self.inception4d_branch1_conv(x_0)
        x_0 = self.inception4d_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception4d_branch2_0_conv(x_1)
        x_1 = self.inception4d_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception4d_branch2_1_conv(x_1)
        x_1 = self.inception4d_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception4d_branch3_0_conv(x_2)
        x_2 = self.inception4d_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception4d_branch3_1_conv(x_2)
        x_2 = self.inception4d_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception4d_branch4_0(x_3)
        x_3 = self.inception4d_branch4_1_conv(x_3)
        x_3 = self.inception4d_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x_0 = x
        x_0 = self.inception4e_branch1_conv(x_0)
        x_0 = self.inception4e_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception4e_branch2_0_conv(x_1)
        x_1 = self.inception4e_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception4e_branch2_1_conv(x_1)
        x_1 = self.inception4e_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception4e_branch3_0_conv(x_2)
        x_2 = self.inception4e_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception4e_branch3_1_conv(x_2)
        x_2 = self.inception4e_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception4e_branch4_0(x_3)
        x_3 = self.inception4e_branch4_1_conv(x_3)
        x_3 = self.inception4e_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x = self.maxpool4(x)
        x_0 = x
        x_0 = self.inception5a_branch1_conv(x_0)
        x_0 = self.inception5a_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception5a_branch2_0_conv(x_1)
        x_1 = self.inception5a_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception5a_branch2_1_conv(x_1)
        x_1 = self.inception5a_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception5a_branch3_0_conv(x_2)
        x_2 = self.inception5a_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception5a_branch3_1_conv(x_2)
        x_2 = self.inception5a_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception5a_branch4_0(x_3)
        x_3 = self.inception5a_branch4_1_conv(x_3)
        x_3 = self.inception5a_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x_0 = x
        x_0 = self.inception5b_branch1_conv(x_0)
        x_0 = self.inception5b_branch1_bn(x_0)
        x_0 = F.relu(x_0, inplace=True)
        x_1 = x
        x_1 = self.inception5b_branch2_0_conv(x_1)
        x_1 = self.inception5b_branch2_0_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_1 = self.inception5b_branch2_1_conv(x_1)
        x_1 = self.inception5b_branch2_1_bn(x_1)
        x_1 = F.relu(x_1, inplace=True)
        x_2 = x
        x_2 = self.inception5b_branch3_0_conv(x_2)
        x_2 = self.inception5b_branch3_0_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_2 = self.inception5b_branch3_1_conv(x_2)
        x_2 = self.inception5b_branch3_1_bn(x_2)
        x_2 = F.relu(x_2, inplace=True)
        x_3 = x
        x_3 = self.inception5b_branch4_0(x_3)
        x_3 = self.inception5b_branch4_1_conv(x_3)
        x_3 = self.inception5b_branch4_1_bn(x_3)
        x_3 = F.relu(x_3, inplace=True)
        x = torch.cat([x_0,x_1,x_2,x_3], 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def googlenet(**kwargs):
    return GoogLeNet(**kwargs)
