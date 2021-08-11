import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 35, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(35, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_0_conv1 = nn.Conv2d(35, 49, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_bn1 = nn.BatchNorm2d(49, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv2 = nn.Conv2d(49, 44, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn2 = nn.BatchNorm2d(44, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv3 = nn.Conv2d(44, 103, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_bn3 = nn.BatchNorm2d(103, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_downsample_0 = nn.Conv2d(35, 103, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_0_downsample_1 = nn.BatchNorm2d(103, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv1 = nn.Conv2d(103, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_1_bn1 = nn.BatchNorm2d(45, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv2 = nn.Conv2d(45, 45, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn2 = nn.BatchNorm2d(45, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv3 = nn.Conv2d(45, 103, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_1_bn3 = nn.BatchNorm2d(103, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv1 = nn.Conv2d(103, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_2_bn1 = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv2 = nn.Conv2d(40, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn2 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv3 = nn.Conv2d(48, 103, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_2_bn3 = nn.BatchNorm2d(103, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv1 = nn.Conv2d(103, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_0_bn1 = nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv2 = nn.Conv2d(88, 88, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer2_0_bn2 = nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv3 = nn.Conv2d(88, 205, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_0_bn3 = nn.BatchNorm2d(205, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_downsample_0 = nn.Conv2d(103, 205, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer2_0_downsample_1 = nn.BatchNorm2d(205, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv1 = nn.Conv2d(205, 91, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_1_bn1 = nn.BatchNorm2d(91, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv2 = nn.Conv2d(91, 92, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn2 = nn.BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv3 = nn.Conv2d(92, 205, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_1_bn3 = nn.BatchNorm2d(205, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv1 = nn.Conv2d(205, 87, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_2_bn1 = nn.BatchNorm2d(87, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv2 = nn.Conv2d(87, 94, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn2 = nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv3 = nn.Conv2d(94, 205, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_2_bn3 = nn.BatchNorm2d(205, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv1 = nn.Conv2d(205, 95, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_3_bn1 = nn.BatchNorm2d(95, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv2 = nn.Conv2d(95, 92, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_3_bn2 = nn.BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_3_conv3 = nn.Conv2d(92, 205, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2_3_bn3 = nn.BatchNorm2d(205, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv1 = nn.Conv2d(205, 179, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_0_bn1 = nn.BatchNorm2d(179, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv2 = nn.Conv2d(179, 183, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer3_0_bn2 = nn.BatchNorm2d(183, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv3 = nn.Conv2d(183, 410, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_0_bn3 = nn.BatchNorm2d(410, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_downsample_0 = nn.Conv2d(205, 410, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer3_0_downsample_1 = nn.BatchNorm2d(410, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv1 = nn.Conv2d(410, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_1_bn1 = nn.BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv2 = nn.Conv2d(184, 162, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn2 = nn.BatchNorm2d(162, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv3 = nn.Conv2d(162, 410, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_1_bn3 = nn.BatchNorm2d(410, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv1 = nn.Conv2d(410, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_2_bn1 = nn.BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv2 = nn.Conv2d(176, 180, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn2 = nn.BatchNorm2d(180, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv3 = nn.Conv2d(180, 410, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_2_bn3 = nn.BatchNorm2d(410, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv1 = nn.Conv2d(410, 182, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_3_bn1 = nn.BatchNorm2d(182, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv2 = nn.Conv2d(182, 181, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_3_bn2 = nn.BatchNorm2d(181, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_3_conv3 = nn.Conv2d(181, 410, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_3_bn3 = nn.BatchNorm2d(410, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv1 = nn.Conv2d(410, 171, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_4_bn1 = nn.BatchNorm2d(171, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv2 = nn.Conv2d(171, 174, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_4_bn2 = nn.BatchNorm2d(174, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_4_conv3 = nn.Conv2d(174, 410, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_4_bn3 = nn.BatchNorm2d(410, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv1 = nn.Conv2d(410, 170, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_5_bn1 = nn.BatchNorm2d(170, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv2 = nn.Conv2d(170, 177, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_5_bn2 = nn.BatchNorm2d(177, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_5_conv3 = nn.Conv2d(177, 410, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_5_bn3 = nn.BatchNorm2d(410, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv1 = nn.Conv2d(410, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_0_bn1 = nn.BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv2 = nn.Conv2d(352, 365, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer4_0_bn2 = nn.BatchNorm2d(365, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_conv3 = nn.Conv2d(365, 820, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_0_bn3 = nn.BatchNorm2d(820, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_0_downsample_0 = nn.Conv2d(410, 820, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer4_0_downsample_1 = nn.BatchNorm2d(820, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv1 = nn.Conv2d(820, 359, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_1_bn1 = nn.BatchNorm2d(359, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv2 = nn.Conv2d(359, 369, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_1_bn2 = nn.BatchNorm2d(369, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_1_conv3 = nn.Conv2d(369, 820, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_1_bn3 = nn.BatchNorm2d(820, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv1 = nn.Conv2d(820, 363, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_2_bn1 = nn.BatchNorm2d(363, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv2 = nn.Conv2d(363, 370, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_2_bn2 = nn.BatchNorm2d(370, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_2_conv3 = nn.Conv2d(370, 820, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_2_bn3 = nn.BatchNorm2d(820, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=820, out_features=1000, bias=True)

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