import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 30, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv1 = nn.Conv2d(30, 26, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn1 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv2 = nn.Conv2d(26, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=26, bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn2 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv3 = nn.Conv2d(26, 11, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn3 = nn.BatchNorm2d(11, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv1 = nn.Conv2d(11, 74, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn1 = nn.BatchNorm2d(74, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv2 = nn.Conv2d(74, 74, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=74, bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn2 = nn.BatchNorm2d(74, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv3 = nn.Conv2d(74, 21, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn3 = nn.BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv1 = nn.Conv2d(21, 114, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn1 = nn.BatchNorm2d(114, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv2 = nn.Conv2d(114, 114, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=114, bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn2 = nn.BatchNorm2d(114, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv3 = nn.Conv2d(114, 21, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn3 = nn.BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv1 = nn.Conv2d(21, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn1 = nn.BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv2 = nn.Conv2d(116, 116, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=116, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn2 = nn.BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv3 = nn.Conv2d(116, 27, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn3 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv1 = nn.Conv2d(27, 155, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn1 = nn.BatchNorm2d(155, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv2 = nn.Conv2d(155, 155, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=155, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn2 = nn.BatchNorm2d(155, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv3 = nn.Conv2d(155, 27, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn3 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv1 = nn.Conv2d(27, 162, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn1 = nn.BatchNorm2d(162, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv2 = nn.Conv2d(162, 162, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=162, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn2 = nn.BatchNorm2d(162, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv3 = nn.Conv2d(162, 27, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn3 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv1 = nn.Conv2d(27, 152, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn1 = nn.BatchNorm2d(152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv2 = nn.Conv2d(152, 152, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=152, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn2 = nn.BatchNorm2d(152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv3 = nn.Conv2d(152, 39, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn3 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv1 = nn.Conv2d(39, 302, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn1 = nn.BatchNorm2d(302, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv2 = nn.Conv2d(302, 302, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=302, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn2 = nn.BatchNorm2d(302, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv3 = nn.Conv2d(302, 39, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn3 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv1 = nn.Conv2d(39, 312, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn1 = nn.BatchNorm2d(312, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv2 = nn.Conv2d(312, 312, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=312, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn2 = nn.BatchNorm2d(312, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv3 = nn.Conv2d(312, 39, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn3 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv1 = nn.Conv2d(39, 307, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn1 = nn.BatchNorm2d(307, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv2 = nn.Conv2d(307, 307, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=307, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn2 = nn.BatchNorm2d(307, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv3 = nn.Conv2d(307, 39, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn3 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv1 = nn.Conv2d(39, 307, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn1 = nn.BatchNorm2d(307, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv2 = nn.Conv2d(307, 307, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=307, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn2 = nn.BatchNorm2d(307, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv3 = nn.Conv2d(307, 77, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn3 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv1 = nn.Conv2d(77, 458, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn1 = nn.BatchNorm2d(458, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv2 = nn.Conv2d(458, 458, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=458, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn2 = nn.BatchNorm2d(458, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv3 = nn.Conv2d(458, 77, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn3 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv1 = nn.Conv2d(77, 461, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn1 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv2 = nn.Conv2d(461, 461, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=461, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn2 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv3 = nn.Conv2d(461, 77, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn3 = nn.BatchNorm2d(77, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv1 = nn.Conv2d(77, 461, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn1 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv2 = nn.Conv2d(461, 461, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=461, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn2 = nn.BatchNorm2d(461, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv3 = nn.Conv2d(461, 126, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn3 = nn.BatchNorm2d(126, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv1 = nn.Conv2d(126, 750, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn1 = nn.BatchNorm2d(750, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv2 = nn.Conv2d(750, 750, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=750, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn2 = nn.BatchNorm2d(750, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv3 = nn.Conv2d(750, 126, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn3 = nn.BatchNorm2d(126, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv1 = nn.Conv2d(126, 778, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn1 = nn.BatchNorm2d(778, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv2 = nn.Conv2d(778, 778, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=778, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn2 = nn.BatchNorm2d(778, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv3 = nn.Conv2d(778, 126, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn3 = nn.BatchNorm2d(126, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv1 = nn.Conv2d(126, 746, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn1 = nn.BatchNorm2d(746, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv2 = nn.Conv2d(746, 746, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=746, bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn2 = nn.BatchNorm2d(746, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv3 = nn.Conv2d(746, 286, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn3 = nn.BatchNorm2d(286, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_last = nn.Conv2d(286, 1150, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_last = nn.BatchNorm2d(1150, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=1150, out_features=1000, bias=True)

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