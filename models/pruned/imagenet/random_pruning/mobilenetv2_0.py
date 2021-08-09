import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv1 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_conv3 = nn.Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_0_LinearBottleneck0_0_bn3 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv1 = nn.Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn2 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_conv3 = nn.Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_0_bn3 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv1 = nn.Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn1 = nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv2 = nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn2 = nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_conv3 = nn.Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_1_LinearBottleneck1_1_bn3 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv1 = nn.Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn1 = nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv2 = nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn2 = nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_conv3 = nn.Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_0_bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv1 = nn.Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn1 = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn2 = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_conv3 = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_1_bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv1 = nn.Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn1 = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn2 = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_conv3 = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_2_LinearBottleneck2_2_bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv1 = nn.Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn1 = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn2 = nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_conv3 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_0_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv1 = nn.Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn1 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv2 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn2 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_conv3 = nn.Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_1_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv1 = nn.Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn1 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv2 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn2 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_conv3 = nn.Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_2_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv1 = nn.Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn1 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv2 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn2 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_conv3 = nn.Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_3_LinearBottleneck3_3_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv1 = nn.Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn1 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv2 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn2 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_conv3 = nn.Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_0_bn3 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv1 = nn.Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn1 = nn.BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv2 = nn.Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn2 = nn.BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_conv3 = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_1_bn3 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv1 = nn.Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn1 = nn.BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv2 = nn.Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn2 = nn.BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_conv3 = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_4_LinearBottleneck4_2_bn3 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv1 = nn.Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn1 = nn.BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv2 = nn.Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn2 = nn.BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_conv3 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_0_bn3 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv1 = nn.Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn1 = nn.BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv2 = nn.Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn2 = nn.BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_conv3 = nn.Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_1_bn3 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv1 = nn.Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn1 = nn.BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv2 = nn.Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn2 = nn.BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_conv3 = nn.Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_5_LinearBottleneck5_2_bn3 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv1 = nn.Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn1 = nn.BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv2 = nn.Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn2 = nn.BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_conv3 = nn.Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bottlenecks_Bottlenecks_6_LinearBottleneck6_0_bn3 = nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_last = nn.Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_last = nn.BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=1280, out_features=1000, bias=True)

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
