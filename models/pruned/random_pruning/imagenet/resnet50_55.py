import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 52, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(52, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_0_conv1 = nn.Conv2d(52, 39, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_bn1 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv2 = nn.Conv2d(39, 46, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn2 = nn.BatchNorm2d(46, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv3 = nn.Conv2d(46, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_bn3 = nn.BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_downsample_0 = nn.Conv2d(52, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_downsample_1 = nn.BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv1 = nn.Conv2d(116, 47, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_1_bn1 = nn.BatchNorm2d(47, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv2 = nn.Conv2d(47, 47, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn2 = nn.BatchNorm2d(47, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv3 = nn.Conv2d(47, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_1_bn3 = nn.BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv1 = nn.Conv2d(116, 46, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_2_bn1 = nn.BatchNorm2d(46, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv2 = nn.Conv2d(46, 53, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn2 = nn.BatchNorm2d(53, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv3 = nn.Conv2d(53, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_2_bn3 = nn.BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv1 = nn.Conv2d(116, 101, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_0_bn1 = nn.BatchNorm2d(101, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv2 = nn.Conv2d(101, 94, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer2_0_bn2 = nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv3 = nn.Conv2d(94, 231, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_0_bn3 = nn.BatchNorm2d(231, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_downsample_0 = nn.Conv2d(116, 231, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer2_0_downsample_1 = nn.BatchNorm2d(231, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv1 = nn.Conv2d(231, 99, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_1_bn1 = nn.BatchNorm2d(99, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv2 = nn.Conv2d(99, 97, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn2 = nn.BatchNorm2d(97, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv3 = nn.Conv2d(97, 231, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_1_bn3 = nn.BatchNorm2d(231, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv1 = nn.Conv2d(231, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_2_bn1 = nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv2 = nn.Conv2d(88, 92, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn2 = nn.BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv3 = nn.Conv2d(92, 231, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_2_bn3 = nn.BatchNorm2d(231, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv1 = nn.Conv2d(231, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_3_bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv2 = nn.Conv2d(96, 94, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_3_bn2 = nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv3 = nn.Conv2d(94, 231, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_3_bn3 = nn.BatchNorm2d(231, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv1 = nn.Conv2d(231, 191, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_0_bn1 = nn.BatchNorm2d(191, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv2 = nn.Conv2d(191, 183, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer3_0_bn2 = nn.BatchNorm2d(183, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv3 = nn.Conv2d(183, 461, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_0_bn3 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_downsample_0 = nn.Conv2d(231, 461, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer3_0_downsample_1 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv1 = nn.Conv2d(461, 189, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_1_bn1 = nn.BatchNorm2d(189, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv2 = nn.Conv2d(189, 181, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn2 = nn.BatchNorm2d(181, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv3 = nn.Conv2d(181, 461, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_1_bn3 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv1 = nn.Conv2d(461, 174, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_2_bn1 = nn.BatchNorm2d(174, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv2 = nn.Conv2d(174, 188, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn2 = nn.BatchNorm2d(188, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv3 = nn.Conv2d(188, 461, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_2_bn3 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv1 = nn.Conv2d(461, 179, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_3_bn1 = nn.BatchNorm2d(179, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv2 = nn.Conv2d(179, 191, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_3_bn2 = nn.BatchNorm2d(191, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv3 = nn.Conv2d(191, 461, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_3_bn3 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv1 = nn.Conv2d(461, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_4_bn1 = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv2 = nn.Conv2d(192, 187, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_4_bn2 = nn.BatchNorm2d(187, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv3 = nn.Conv2d(187, 461, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_4_bn3 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv1 = nn.Conv2d(461, 197, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_5_bn1 = nn.BatchNorm2d(197, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv2 = nn.Conv2d(197, 189, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_5_bn2 = nn.BatchNorm2d(189, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv3 = nn.Conv2d(189, 461, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_5_bn3 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv1 = nn.Conv2d(461, 379, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_0_bn1 = nn.BatchNorm2d(379, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv2 = nn.Conv2d(379, 375, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer4_0_bn2 = nn.BatchNorm2d(375, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv3 = nn.Conv2d(375, 922, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_0_bn3 = nn.BatchNorm2d(922, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_downsample_0 = nn.Conv2d(461, 922, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer4_0_downsample_1 = nn.BatchNorm2d(922, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv1 = nn.Conv2d(922, 390, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_1_bn1 = nn.BatchNorm2d(390, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv2 = nn.Conv2d(390, 382, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_1_bn2 = nn.BatchNorm2d(382, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv3 = nn.Conv2d(382, 922, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_1_bn3 = nn.BatchNorm2d(922, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv1 = nn.Conv2d(922, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_2_bn1 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv2 = nn.Conv2d(384, 367, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_2_bn2 = nn.BatchNorm2d(367, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv3 = nn.Conv2d(367, 922, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_2_bn3 = nn.BatchNorm2d(922, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=922, out_features=1000, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        x_main = x
        x_main = self.layer1_0_conv1(x_main)
        x_main = self.layer1_0_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer1_0_conv2(x_main)
        x_main = self.layer1_0_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer1_0_conv3(x_main)
        x_main = self.layer1_0_bn3(x_main)
        x_residual = x
        x_residual = self.layer1_0_downsample_0(x_residual)
        x_residual = self.layer1_0_downsample_1(x_residual)
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer1_1_conv1(x_main)
        x_main = self.layer1_1_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer1_1_conv2(x_main)
        x_main = self.layer1_1_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer1_1_conv3(x_main)
        x_main = self.layer1_1_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer1_2_conv1(x_main)
        x_main = self.layer1_2_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer1_2_conv2(x_main)
        x_main = self.layer1_2_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer1_2_conv3(x_main)
        x_main = self.layer1_2_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer2_0_conv1(x_main)
        x_main = self.layer2_0_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_0_conv2(x_main)
        x_main = self.layer2_0_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_0_conv3(x_main)
        x_main = self.layer2_0_bn3(x_main)
        x_residual = x
        x_residual = self.layer2_0_downsample_0(x_residual)
        x_residual = self.layer2_0_downsample_1(x_residual)
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer2_1_conv1(x_main)
        x_main = self.layer2_1_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_1_conv2(x_main)
        x_main = self.layer2_1_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_1_conv3(x_main)
        x_main = self.layer2_1_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer2_2_conv1(x_main)
        x_main = self.layer2_2_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_2_conv2(x_main)
        x_main = self.layer2_2_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_2_conv3(x_main)
        x_main = self.layer2_2_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer2_3_conv1(x_main)
        x_main = self.layer2_3_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_3_conv2(x_main)
        x_main = self.layer2_3_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_3_conv3(x_main)
        x_main = self.layer2_3_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer3_0_conv1(x_main)
        x_main = self.layer3_0_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_0_conv2(x_main)
        x_main = self.layer3_0_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_0_conv3(x_main)
        x_main = self.layer3_0_bn3(x_main)
        x_residual = x
        x_residual = self.layer3_0_downsample_0(x_residual)
        x_residual = self.layer3_0_downsample_1(x_residual)
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer3_1_conv1(x_main)
        x_main = self.layer3_1_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_1_conv2(x_main)
        x_main = self.layer3_1_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_1_conv3(x_main)
        x_main = self.layer3_1_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer3_2_conv1(x_main)
        x_main = self.layer3_2_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_2_conv2(x_main)
        x_main = self.layer3_2_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_2_conv3(x_main)
        x_main = self.layer3_2_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer3_3_conv1(x_main)
        x_main = self.layer3_3_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_3_conv2(x_main)
        x_main = self.layer3_3_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_3_conv3(x_main)
        x_main = self.layer3_3_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer3_4_conv1(x_main)
        x_main = self.layer3_4_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_4_conv2(x_main)
        x_main = self.layer3_4_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_4_conv3(x_main)
        x_main = self.layer3_4_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer3_5_conv1(x_main)
        x_main = self.layer3_5_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_5_conv2(x_main)
        x_main = self.layer3_5_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_5_conv3(x_main)
        x_main = self.layer3_5_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer4_0_conv1(x_main)
        x_main = self.layer4_0_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer4_0_conv2(x_main)
        x_main = self.layer4_0_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer4_0_conv3(x_main)
        x_main = self.layer4_0_bn3(x_main)
        x_residual = x
        x_residual = self.layer4_0_downsample_0(x_residual)
        x_residual = self.layer4_0_downsample_1(x_residual)
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer4_1_conv1(x_main)
        x_main = self.layer4_1_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer4_1_conv2(x_main)
        x_main = self.layer4_1_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer4_1_conv3(x_main)
        x_main = self.layer4_1_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer4_2_conv1(x_main)
        x_main = self.layer4_2_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer4_2_conv2(x_main)
        x_main = self.layer4_2_bn2(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer4_2_conv3(x_main)
        x_main = self.layer4_2_bn3(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50(**kwargs):
    return ResNet50(**kwargs)