import torch
import torch.nn as nn
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1_conv = nn.Conv2d(3, 31, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1_bn = nn.BatchNorm2d(31, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_conv = nn.Conv2d(31, 30, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2_bn = nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_conv = nn.Conv2d(30, 82, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_bn = nn.BatchNorm2d(82, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception3a_branch1_conv = nn.Conv2d(82, 30, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch1_bn = nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch2_0_conv = nn.Conv2d(82, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch2_0_bn = nn.BatchNorm2d(45, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch2_1_conv = nn.Conv2d(45, 49, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3a_branch2_1_bn = nn.BatchNorm2d(49, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch3_0_conv = nn.Conv2d(82, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch3_0_bn = nn.BatchNorm2d(8, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch3_1_conv = nn.Conv2d(8, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3a_branch3_1_bn = nn.BatchNorm2d(10, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception3a_branch4_1_conv = nn.Conv2d(82, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3a_branch4_1_bn = nn.BatchNorm2d(15, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch1_conv = nn.Conv2d(104, 71, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch1_bn = nn.BatchNorm2d(71, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch2_0_conv = nn.Conv2d(104, 61, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch2_0_bn = nn.BatchNorm2d(61, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch2_1_conv = nn.Conv2d(61, 71, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3b_branch2_1_bn = nn.BatchNorm2d(71, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch3_0_conv = nn.Conv2d(104, 11, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch3_0_bn = nn.BatchNorm2d(11, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch3_1_conv = nn.Conv2d(11, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception3b_branch3_1_bn = nn.BatchNorm2d(41, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception3b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception3b_branch4_1_conv = nn.Conv2d(104, 35, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception3b_branch4_1_bn = nn.BatchNorm2d(35, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception4a_branch1_conv = nn.Conv2d(218, 76, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch1_bn = nn.BatchNorm2d(76, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch2_0_conv = nn.Conv2d(218, 38, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch2_0_bn = nn.BatchNorm2d(38, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch2_1_conv = nn.Conv2d(38, 107, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4a_branch2_1_bn = nn.BatchNorm2d(107, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch3_0_conv = nn.Conv2d(218, 9, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch3_0_bn = nn.BatchNorm2d(9, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch3_1_conv = nn.Conv2d(9, 25, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4a_branch3_1_bn = nn.BatchNorm2d(25, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4a_branch4_1_conv = nn.Conv2d(218, 33, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4a_branch4_1_bn = nn.BatchNorm2d(33, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch1_conv = nn.Conv2d(241, 73, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch1_bn = nn.BatchNorm2d(73, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch2_0_conv = nn.Conv2d(241, 46, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch2_0_bn = nn.BatchNorm2d(46, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch2_1_conv = nn.Conv2d(46, 113, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4b_branch2_1_bn = nn.BatchNorm2d(113, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch3_0_conv = nn.Conv2d(241, 13, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch3_0_bn = nn.BatchNorm2d(13, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch3_1_conv = nn.Conv2d(13, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4b_branch3_1_bn = nn.BatchNorm2d(26, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4b_branch4_1_conv = nn.Conv2d(241, 35, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4b_branch4_1_bn = nn.BatchNorm2d(35, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch1_conv = nn.Conv2d(247, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch1_bn = nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch2_0_conv = nn.Conv2d(247, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch2_0_bn = nn.BatchNorm2d(58, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch2_1_conv = nn.Conv2d(58, 130, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4c_branch2_1_bn = nn.BatchNorm2d(130, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch3_0_conv = nn.Conv2d(247, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch3_0_bn = nn.BatchNorm2d(10, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch3_1_conv = nn.Conv2d(10, 31, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4c_branch3_1_bn = nn.BatchNorm2d(31, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4c_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4c_branch4_1_conv = nn.Conv2d(247, 34, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4c_branch4_1_bn = nn.BatchNorm2d(34, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch1_conv = nn.Conv2d(255, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch1_bn = nn.BatchNorm2d(58, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch2_0_conv = nn.Conv2d(255, 67, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch2_0_bn = nn.BatchNorm2d(67, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch2_1_conv = nn.Conv2d(67, 133, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4d_branch2_1_bn = nn.BatchNorm2d(133, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch3_0_conv = nn.Conv2d(255, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch3_0_bn = nn.BatchNorm2d(18, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch3_1_conv = nn.Conv2d(18, 35, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4d_branch3_1_bn = nn.BatchNorm2d(35, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4d_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4d_branch4_1_conv = nn.Conv2d(255, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4d_branch4_1_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch1_conv = nn.Conv2d(258, 121, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch1_bn = nn.BatchNorm2d(121, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch2_0_conv = nn.Conv2d(258, 65, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch2_0_bn = nn.BatchNorm2d(65, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch2_1_conv = nn.Conv2d(65, 141, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4e_branch2_1_bn = nn.BatchNorm2d(141, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch3_0_conv = nn.Conv2d(258, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch3_0_bn = nn.BatchNorm2d(17, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch3_1_conv = nn.Conv2d(17, 53, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception4e_branch3_1_bn = nn.BatchNorm2d(53, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception4e_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception4e_branch4_1_conv = nn.Conv2d(258, 63, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception4e_branch4_1_bn = nn.BatchNorm2d(63, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.inception5a_branch1_conv = nn.Conv2d(378, 108, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch1_bn = nn.BatchNorm2d(108, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch2_0_conv = nn.Conv2d(378, 82, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch2_0_bn = nn.BatchNorm2d(82, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch2_1_conv = nn.Conv2d(82, 152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5a_branch2_1_bn = nn.BatchNorm2d(152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch3_0_conv = nn.Conv2d(378, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch3_0_bn = nn.BatchNorm2d(20, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch3_1_conv = nn.Conv2d(20, 57, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5a_branch3_1_bn = nn.BatchNorm2d(57, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5a_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception5a_branch4_1_conv = nn.Conv2d(378, 50, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5a_branch4_1_bn = nn.BatchNorm2d(50, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch1_conv = nn.Conv2d(367, 173, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch1_bn = nn.BatchNorm2d(173, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch2_0_conv = nn.Conv2d(367, 87, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch2_0_bn = nn.BatchNorm2d(87, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch2_1_conv = nn.Conv2d(87, 178, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5b_branch2_1_bn = nn.BatchNorm2d(178, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch3_0_conv = nn.Conv2d(367, 23, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch3_0_bn = nn.BatchNorm2d(23, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch3_1_conv = nn.Conv2d(23, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.inception5b_branch3_1_bn = nn.BatchNorm2d(58, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.inception5b_branch4_0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.inception5b_branch4_1_conv = nn.Conv2d(367, 54, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.inception5b_branch4_1_bn = nn.BatchNorm2d(54, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=463, out_features=1000, bias=True)

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
