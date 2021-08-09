import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 9, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv1 = nn.Conv2d(9, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn1 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv2 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2, bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn2 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv3 = nn.Conv2d(2, 5, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn3 = nn.BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv1 = nn.Conv2d(5, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn1 = nn.BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv2 = nn.Conv2d(17, 17, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=17, bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn2 = nn.BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv3 = nn.Conv2d(17, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn3 = nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv1 = nn.Conv2d(4, 22, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn1 = nn.BatchNorm2d(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv2 = nn.Conv2d(22, 22, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=22, bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn2 = nn.BatchNorm2d(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv3 = nn.Conv2d(22, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn3 = nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv1 = nn.Conv2d(4, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn1 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv2 = nn.Conv2d(18, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=18, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn2 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv3 = nn.Conv2d(18, 5, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn3 = nn.BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv1 = nn.Conv2d(5, 30, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn1 = nn.BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv2 = nn.Conv2d(30, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=30, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn2 = nn.BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv3 = nn.Conv2d(30, 5, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn3 = nn.BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv1 = nn.Conv2d(5, 35, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn1 = nn.BatchNorm2d(35, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv2 = nn.Conv2d(35, 35, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=35, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn2 = nn.BatchNorm2d(35, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv3 = nn.Conv2d(35, 5, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn3 = nn.BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv1 = nn.Conv2d(5, 27, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn1 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv2 = nn.Conv2d(27, 27, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=27, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn2 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv3 = nn.Conv2d(27, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn3 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv1 = nn.Conv2d(10, 71, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn1 = nn.BatchNorm2d(71, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv2 = nn.Conv2d(71, 71, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=71, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn2 = nn.BatchNorm2d(71, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv3 = nn.Conv2d(71, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn3 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv1 = nn.Conv2d(10, 57, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn1 = nn.BatchNorm2d(57, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv2 = nn.Conv2d(57, 57, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=57, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn2 = nn.BatchNorm2d(57, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv3 = nn.Conv2d(57, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn3 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv1 = nn.Conv2d(10, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn1 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv2 = nn.Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn2 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv3 = nn.Conv2d(48, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn3 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv1 = nn.Conv2d(10, 52, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn1 = nn.BatchNorm2d(52, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv2 = nn.Conv2d(52, 52, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=52, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn2 = nn.BatchNorm2d(52, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv3 = nn.Conv2d(52, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn3 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv1 = nn.Conv2d(15, 108, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn1 = nn.BatchNorm2d(108, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv2 = nn.Conv2d(108, 108, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=108, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn2 = nn.BatchNorm2d(108, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv3 = nn.Conv2d(108, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn3 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv1 = nn.Conv2d(15, 87, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn1 = nn.BatchNorm2d(87, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv2 = nn.Conv2d(87, 87, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=87, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn2 = nn.BatchNorm2d(87, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv3 = nn.Conv2d(87, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn3 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv1 = nn.Conv2d(15, 91, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn1 = nn.BatchNorm2d(91, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv2 = nn.Conv2d(91, 91, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=91, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn2 = nn.BatchNorm2d(91, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv3 = nn.Conv2d(91, 25, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn3 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv1 = nn.Conv2d(25, 140, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn1 = nn.BatchNorm2d(140, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv2 = nn.Conv2d(140, 140, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=140, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn2 = nn.BatchNorm2d(140, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv3 = nn.Conv2d(140, 25, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn3 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv1 = nn.Conv2d(25, 154, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn1 = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv2 = nn.Conv2d(154, 154, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=154, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn2 = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv3 = nn.Conv2d(154, 25, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn3 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv1 = nn.Conv2d(25, 132, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn1 = nn.BatchNorm2d(132, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv2 = nn.Conv2d(132, 132, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=132, bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn2 = nn.BatchNorm2d(132, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv3 = nn.Conv2d(132, 134, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn3 = nn.BatchNorm2d(134, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_last = nn.Conv2d(134, 501, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_last = nn.BatchNorm2d(501, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=501, out_features=1000, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv1(x)
        x = self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv2(x)
        x = self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv3(x)
        x = self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn3(x)
        x = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv1(x)
        x = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv2(x)
        x = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv3(x)
        x = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn3(x)
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv1(x)
        x = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv2(x)
        x = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv3(x)
        x = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn3(x)
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv1(x)
        x = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv2(x)
        x = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv3(x)
        x = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn3(x)
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv1(x)
        x = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv2(x)
        x = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv3(x)
        x = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn3(x)
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv1(x)
        x = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv2(x)
        x = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv3(x)
        x = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn3(x)
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv1(x_main)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv2(x_main)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn2(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv3(x_main)
        x_main = self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv1(x)
        x = self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv2(x)
        x = self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn2(x)
        x = F.relu6(x, inplace=True)
        x = self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv3(x)
        x = self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn3(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = F.relu6(x, inplace=True)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)
