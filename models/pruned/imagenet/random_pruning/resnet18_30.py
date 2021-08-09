import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 43, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_0_conv1 = nn.Conv2d(43, 54, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn1 = nn.BatchNorm2d(54, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv2 = nn.Conv2d(54, 43, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn2 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv1 = nn.Conv2d(43, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn1 = nn.BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv2 = nn.Conv2d(58, 43, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn2 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv1 = nn.Conv2d(43, 116, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer2_0_bn1 = nn.BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv2 = nn.Conv2d(116, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_0_bn2 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_downsample_0 = nn.Conv2d(43, 100, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer2_0_downsample_1 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv1 = nn.Conv2d(100, 113, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn1 = nn.BatchNorm2d(113, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv2 = nn.Conv2d(113, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn2 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv1 = nn.Conv2d(100, 228, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer3_0_bn1 = nn.BatchNorm2d(228, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv2 = nn.Conv2d(228, 202, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_0_bn2 = nn.BatchNorm2d(202, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_downsample_0 = nn.Conv2d(100, 202, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer3_0_downsample_1 = nn.BatchNorm2d(202, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv1 = nn.Conv2d(202, 225, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn1 = nn.BatchNorm2d(225, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv2 = nn.Conv2d(225, 202, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn2 = nn.BatchNorm2d(202, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv1 = nn.Conv2d(202, 455, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer4_0_bn1 = nn.BatchNorm2d(455, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv2 = nn.Conv2d(455, 397, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_0_bn2 = nn.BatchNorm2d(397, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_downsample_0 = nn.Conv2d(202, 397, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer4_0_downsample_1 = nn.BatchNorm2d(397, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv1 = nn.Conv2d(397, 462, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_1_bn1 = nn.BatchNorm2d(462, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv2 = nn.Conv2d(462, 397, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_1_bn2 = nn.BatchNorm2d(397, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=397, out_features=1000, bias=True)

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
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer1_1_conv1(x_main)
        x_main = self.layer1_1_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer1_1_conv2(x_main)
        x_main = self.layer1_1_bn2(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer2_0_conv1(x_main)
        x_main = self.layer2_0_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer2_0_conv2(x_main)
        x_main = self.layer2_0_bn2(x_main)
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
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer3_0_conv1(x_main)
        x_main = self.layer3_0_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer3_0_conv2(x_main)
        x_main = self.layer3_0_bn2(x_main)
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
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x_main = x
        x_main = self.layer4_0_conv1(x_main)
        x_main = self.layer4_0_bn1(x_main)
        x_main = F.relu(x_main, inplace=True)
        x_main = self.layer4_0_conv2(x_main)
        x_main = self.layer4_0_bn2(x_main)
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
        x_residual = x
        x = F.relu(x_main + x_residual, inplace=True)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18(**kwargs):
    return ResNet18(**kwargs)
