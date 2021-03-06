import torch
import torch.nn as nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1_conv = nn.Conv2d(3, 55, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1_bn = nn.BatchNorm2d(55, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_conv = nn.Conv2d(55, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2_bn = nn.BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_conv = nn.Conv2d(48, 157, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_bn = nn.BatchNorm2d(157, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception3a_branch1_conv = nn.Conv2d(157, 53, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch1_bn = nn.BatchNorm2d(53, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch2_0_conv = nn.Conv2d(157, 82, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch2_0_bn = nn.BatchNorm2d(82, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch2_1_conv = nn.Conv2d(82, 103, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3a_branch2_1_bn = nn.BatchNorm2d(103, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch3_0_conv = nn.Conv2d(157, 11, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch3_0_bn = nn.BatchNorm2d(11, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch3_1_conv = nn.Conv2d(11, 25, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3a_branch3_1_bn = nn.BatchNorm2d(25, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception3a_branch4_1_conv = nn.Conv2d(157, 28, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch4_1_bn = nn.BatchNorm2d(28, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch1_conv = nn.Conv2d(209, 109, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch1_bn = nn.BatchNorm2d(109, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch2_0_conv = nn.Conv2d(209, 110, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch2_0_bn = nn.BatchNorm2d(110, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch2_1_conv = nn.Conv2d(110, 161, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3b_branch2_1_bn = nn.BatchNorm2d(161, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch3_0_conv = nn.Conv2d(209, 21, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch3_0_bn = nn.BatchNorm2d(21, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch3_1_conv = nn.Conv2d(21, 79, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3b_branch3_1_bn = nn.BatchNorm2d(79, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception3b_branch4_1_conv = nn.Conv2d(209, 52, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch4_1_bn = nn.BatchNorm2d(52, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception4a_branch1_conv = nn.Conv2d(401, 158, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch1_bn = nn.BatchNorm2d(158, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch2_0_conv = nn.Conv2d(401, 75, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch2_0_bn = nn.BatchNorm2d(75, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch2_1_conv = nn.Conv2d(75, 166, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4a_branch2_1_bn = nn.BatchNorm2d(166, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch3_0_conv = nn.Conv2d(401, 14, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch3_0_bn = nn.BatchNorm2d(14, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch3_1_conv = nn.Conv2d(14, 44, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4a_branch3_1_bn = nn.BatchNorm2d(44, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4a_branch4_1_conv = nn.Conv2d(401, 59, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch4_1_bn = nn.BatchNorm2d(59, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch1_conv = nn.Conv2d(427, 134, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch1_bn = nn.BatchNorm2d(134, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch2_0_conv = nn.Conv2d(427, 92, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch2_0_bn = nn.BatchNorm2d(92, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch2_1_conv = nn.Conv2d(92, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4b_branch2_1_bn = nn.BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch3_0_conv = nn.Conv2d(427, 21, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch3_0_bn = nn.BatchNorm2d(21, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch3_1_conv = nn.Conv2d(21, 53, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4b_branch3_1_bn = nn.BatchNorm2d(53, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4b_branch4_1_conv = nn.Conv2d(427, 59, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch4_1_bn = nn.BatchNorm2d(59, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch1_conv = nn.Conv2d(438, 106, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch1_bn = nn.BatchNorm2d(106, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch2_0_conv = nn.Conv2d(438, 108, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch2_0_bn = nn.BatchNorm2d(108, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch2_1_conv = nn.Conv2d(108, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4c_branch2_1_bn = nn.BatchNorm2d(208, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch3_0_conv = nn.Conv2d(438, 19, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch3_0_bn = nn.BatchNorm2d(19, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch3_1_conv = nn.Conv2d(19, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4c_branch3_1_bn = nn.BatchNorm2d(50, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4c_branch4_1_conv = nn.Conv2d(438, 51, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch4_1_bn = nn.BatchNorm2d(51, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch1_conv = nn.Conv2d(415, 95, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch1_bn = nn.BatchNorm2d(95, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch2_0_conv = nn.Conv2d(415, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch2_0_bn = nn.BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch2_1_conv = nn.Conv2d(112, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4d_branch2_1_bn = nn.BatchNorm2d(232, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch3_0_conv = nn.Conv2d(415, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch3_0_bn = nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch3_1_conv = nn.Conv2d(24, 52, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4d_branch3_1_bn = nn.BatchNorm2d(52, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4d_branch4_1_conv = nn.Conv2d(415, 50, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch4_1_bn = nn.BatchNorm2d(50, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch1_conv = nn.Conv2d(429, 214, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch1_bn = nn.BatchNorm2d(214, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch2_0_conv = nn.Conv2d(429, 138, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch2_0_bn = nn.BatchNorm2d(138, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch2_1_conv = nn.Conv2d(138, 272, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4e_branch2_1_bn = nn.BatchNorm2d(272, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch3_0_conv = nn.Conv2d(429, 27, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch3_0_bn = nn.BatchNorm2d(27, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch3_1_conv = nn.Conv2d(27, 107, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4e_branch3_1_bn = nn.BatchNorm2d(107, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4e_branch4_1_conv = nn.Conv2d(429, 108, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch4_1_bn = nn.BatchNorm2d(108, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception5a_branch1_conv = nn.Conv2d(701, 209, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch1_bn = nn.BatchNorm2d(209, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch2_0_conv = nn.Conv2d(701, 125, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch2_0_bn = nn.BatchNorm2d(125, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch2_1_conv = nn.Conv2d(125, 265, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5a_branch2_1_bn = nn.BatchNorm2d(265, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch3_0_conv = nn.Conv2d(701, 28, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch3_0_bn = nn.BatchNorm2d(28, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch3_1_conv = nn.Conv2d(28, 107, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5a_branch3_1_bn = nn.BatchNorm2d(107, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception5a_branch4_1_conv = nn.Conv2d(701, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch4_1_bn = nn.BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch1_conv = nn.Conv2d(693, 299, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch1_bn = nn.BatchNorm2d(299, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch2_0_conv = nn.Conv2d(693, 162, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch2_0_bn = nn.BatchNorm2d(162, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch2_1_conv = nn.Conv2d(162, 314, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5b_branch2_1_bn = nn.BatchNorm2d(314, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch3_0_conv = nn.Conv2d(693, 35, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch3_0_bn = nn.BatchNorm2d(35, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch3_1_conv = nn.Conv2d(35, 111, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5b_branch3_1_bn = nn.BatchNorm2d(111, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception5b_branch4_1_conv = nn.Conv2d(693, 103, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch4_1_bn = nn.BatchNorm2d(103, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=827, out_features=1000, bias=True)

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
