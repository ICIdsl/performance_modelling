import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 41, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(41, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_0_conv1 = nn.Conv2d(41, 37, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_bn1 = nn.BatchNorm2d(37, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv2 = nn.Conv2d(37, 43, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn2 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv3 = nn.Conv2d(43, 68, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_bn3 = nn.BatchNorm2d(68, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_downsample_0 = nn.Conv2d(41, 68, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_downsample_1 = nn.BatchNorm2d(68, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv1 = nn.Conv2d(68, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_1_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv2 = nn.Conv2d(32, 42, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn2 = nn.BatchNorm2d(42, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv3 = nn.Conv2d(42, 68, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_1_bn3 = nn.BatchNorm2d(68, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv1 = nn.Conv2d(68, 41, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_2_bn1 = nn.BatchNorm2d(41, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv2 = nn.Conv2d(41, 42, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn2 = nn.BatchNorm2d(42, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv3 = nn.Conv2d(42, 68, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_2_bn3 = nn.BatchNorm2d(68, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv1 = nn.Conv2d(68, 81, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_0_bn1 = nn.BatchNorm2d(81, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv2 = nn.Conv2d(81, 83, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer2_0_bn2 = nn.BatchNorm2d(83, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv3 = nn.Conv2d(83, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_0_bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_downsample_0 = nn.Conv2d(68, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer2_0_downsample_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv1 = nn.Conv2d(128, 76, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_1_bn1 = nn.BatchNorm2d(76, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv2 = nn.Conv2d(76, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn2 = nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv3 = nn.Conv2d(88, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_1_bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv1 = nn.Conv2d(128, 82, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_2_bn1 = nn.BatchNorm2d(82, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv2 = nn.Conv2d(82, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn2 = nn.BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv3 = nn.Conv2d(80, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_2_bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv1 = nn.Conv2d(128, 69, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_3_bn1 = nn.BatchNorm2d(69, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv2 = nn.Conv2d(69, 82, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_3_bn2 = nn.BatchNorm2d(82, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv3 = nn.Conv2d(82, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_3_bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv1 = nn.Conv2d(128, 177, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_0_bn1 = nn.BatchNorm2d(177, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv2 = nn.Conv2d(177, 150, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer3_0_bn2 = nn.BatchNorm2d(150, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv3 = nn.Conv2d(150, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_0_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_downsample_0 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer3_0_downsample_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv1 = nn.Conv2d(256, 146, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_1_bn1 = nn.BatchNorm2d(146, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv2 = nn.Conv2d(146, 154, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn2 = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv3 = nn.Conv2d(154, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_1_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv1 = nn.Conv2d(256, 157, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_2_bn1 = nn.BatchNorm2d(157, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv2 = nn.Conv2d(157, 147, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn2 = nn.BatchNorm2d(147, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv3 = nn.Conv2d(147, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_2_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv1 = nn.Conv2d(256, 155, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_3_bn1 = nn.BatchNorm2d(155, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv2 = nn.Conv2d(155, 159, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_3_bn2 = nn.BatchNorm2d(159, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv3 = nn.Conv2d(159, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_3_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv1 = nn.Conv2d(256, 147, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_4_bn1 = nn.BatchNorm2d(147, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv2 = nn.Conv2d(147, 158, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_4_bn2 = nn.BatchNorm2d(158, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv3 = nn.Conv2d(158, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_4_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv1 = nn.Conv2d(256, 156, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_5_bn1 = nn.BatchNorm2d(156, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv2 = nn.Conv2d(156, 157, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_5_bn2 = nn.BatchNorm2d(157, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv3 = nn.Conv2d(157, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_5_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv1 = nn.Conv2d(256, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_0_bn1 = nn.BatchNorm2d(304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv2 = nn.Conv2d(304, 312, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer4_0_bn2 = nn.BatchNorm2d(312, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv3 = nn.Conv2d(312, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_0_bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_downsample_0 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer4_0_downsample_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv1 = nn.Conv2d(512, 321, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_1_bn1 = nn.BatchNorm2d(321, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv2 = nn.Conv2d(321, 309, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_1_bn2 = nn.BatchNorm2d(309, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv3 = nn.Conv2d(309, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_1_bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv1 = nn.Conv2d(512, 323, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_2_bn1 = nn.BatchNorm2d(323, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv2 = nn.Conv2d(323, 329, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_2_bn2 = nn.BatchNorm2d(329, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv3 = nn.Conv2d(329, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_2_bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=1000, bias=True)

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
