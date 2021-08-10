import torch
import torch.nn as nn
import torch.nn.functional as F

class MnasNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layers_0 = nn.Conv2d(3, 14, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layers_1 = nn.BatchNorm2d(14, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_3 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=14, bias=False)
        self.layers_4 = nn.BatchNorm2d(14, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_6 = nn.Conv2d(14, 7, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_7 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_0 = nn.Conv2d(7, 11, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_1 = nn.BatchNorm2d(11, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_3 = nn.Conv2d(11, 11, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=11, bias=False)
        self.layers_8_0_layers_4 = nn.BatchNorm2d(11, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_6 = nn.Conv2d(11, 7, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_7 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_0 = nn.Conv2d(7, 23, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_1 = nn.BatchNorm2d(23, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_3 = nn.Conv2d(23, 23, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=23, bias=False)
        self.layers_8_1_layers_4 = nn.BatchNorm2d(23, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_6 = nn.Conv2d(23, 7, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_7 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_0 = nn.Conv2d(7, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_2_layers_1 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_3 = nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)
        self.layers_8_2_layers_4 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_6 = nn.Conv2d(20, 7, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_2_layers_7 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_0 = nn.Conv2d(7, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_1 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_3 = nn.Conv2d(24, 24, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=24, bias=False)
        self.layers_9_0_layers_4 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_6 = nn.Conv2d(24, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_7 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_0 = nn.Conv2d(10, 33, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_1 = nn.BatchNorm2d(33, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_3 = nn.Conv2d(33, 33, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=33, bias=False)
        self.layers_9_1_layers_4 = nn.BatchNorm2d(33, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_6 = nn.Conv2d(33, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_7 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_0 = nn.Conv2d(10, 43, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_1 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_3 = nn.Conv2d(43, 43, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=43, bias=False)
        self.layers_9_2_layers_4 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_6 = nn.Conv2d(43, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_7 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_0 = nn.Conv2d(10, 83, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_1 = nn.BatchNorm2d(83, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_3 = nn.Conv2d(83, 83, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=83, bias=False)
        self.layers_10_0_layers_4 = nn.BatchNorm2d(83, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_6 = nn.Conv2d(83, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_7 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_0 = nn.Conv2d(20, 137, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_1 = nn.BatchNorm2d(137, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_3 = nn.Conv2d(137, 137, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=137, bias=False)
        self.layers_10_1_layers_4 = nn.BatchNorm2d(137, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_6 = nn.Conv2d(137, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_7 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_0 = nn.Conv2d(20, 167, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_1 = nn.BatchNorm2d(167, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_3 = nn.Conv2d(167, 167, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=167, bias=False)
        self.layers_10_2_layers_4 = nn.BatchNorm2d(167, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_6 = nn.Conv2d(167, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_7 = nn.BatchNorm2d(20, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_0 = nn.Conv2d(20, 156, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_1 = nn.BatchNorm2d(156, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_3 = nn.Conv2d(156, 156, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=156, bias=False)
        self.layers_11_0_layers_4 = nn.BatchNorm2d(156, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_6 = nn.Conv2d(156, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_7 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_0 = nn.Conv2d(24, 186, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_1 = nn.BatchNorm2d(186, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_3 = nn.Conv2d(186, 186, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=186, bias=False)
        self.layers_11_1_layers_4 = nn.BatchNorm2d(186, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_6 = nn.Conv2d(186, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_7 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_0 = nn.Conv2d(24, 210, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_1 = nn.BatchNorm2d(210, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_3 = nn.Conv2d(210, 210, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=210, bias=False)
        self.layers_12_0_layers_4 = nn.BatchNorm2d(210, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_6 = nn.Conv2d(210, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_7 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_0 = nn.Conv2d(48, 392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_1 = nn.BatchNorm2d(392, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_3 = nn.Conv2d(392, 392, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=392, bias=False)
        self.layers_12_1_layers_4 = nn.BatchNorm2d(392, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_6 = nn.Conv2d(392, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_7 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_0 = nn.Conv2d(48, 374, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_1 = nn.BatchNorm2d(374, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_3 = nn.Conv2d(374, 374, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=374, bias=False)
        self.layers_12_2_layers_4 = nn.BatchNorm2d(374, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_6 = nn.Conv2d(374, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_7 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_0 = nn.Conv2d(48, 346, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_1 = nn.BatchNorm2d(346, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_3 = nn.Conv2d(346, 346, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=346, bias=False)
        self.layers_12_3_layers_4 = nn.BatchNorm2d(346, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_6 = nn.Conv2d(346, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_7 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_0 = nn.Conv2d(48, 361, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_1 = nn.BatchNorm2d(361, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_3 = nn.Conv2d(361, 361, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=361, bias=False)
        self.layers_13_0_layers_4 = nn.BatchNorm2d(361, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_6 = nn.Conv2d(361, 190, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_7 = nn.BatchNorm2d(190, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_14 = nn.Conv2d(190, 732, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_15 = nn.BatchNorm2d(732, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier_1 = nn.Linear(in_features=732, out_features=1000, bias=True)

    def forward(self, x):
        x = self.layers_0(x)
        x = self.layers_1(x)
        x = F.relu(x, inplace=True)
        x = self.layers_3(x)
        x = self.layers_4(x)
        x = F.relu(x, inplace=True)
        x = self.layers_6(x)
        x = self.layers_7(x)
        x = self.layers_8_0_layers_0(x)
        x = self.layers_8_0_layers_1(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_8_0_layers_3(x)
        x = self.layers_8_0_layers_4(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_8_0_layers_6(x)
        x = self.layers_8_0_layers_7(x)
        x_main = x
        x_main = self.layers_8_1_layers_0(x_main)
        x_main = self.layers_8_1_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_8_1_layers_3(x_main)
        x_main = self.layers_8_1_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_8_1_layers_6(x_main)
        x_main = self.layers_8_1_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_8_2_layers_0(x_main)
        x_main = self.layers_8_2_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_8_2_layers_3(x_main)
        x_main = self.layers_8_2_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_8_2_layers_6(x_main)
        x_main = self.layers_8_2_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.layers_9_0_layers_0(x)
        x = self.layers_9_0_layers_1(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_9_0_layers_3(x)
        x = self.layers_9_0_layers_4(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_9_0_layers_6(x)
        x = self.layers_9_0_layers_7(x)
        x_main = x
        x_main = self.layers_9_1_layers_0(x_main)
        x_main = self.layers_9_1_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_9_1_layers_3(x_main)
        x_main = self.layers_9_1_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_9_1_layers_6(x_main)
        x_main = self.layers_9_1_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_9_2_layers_0(x_main)
        x_main = self.layers_9_2_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_9_2_layers_3(x_main)
        x_main = self.layers_9_2_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_9_2_layers_6(x_main)
        x_main = self.layers_9_2_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.layers_10_0_layers_0(x)
        x = self.layers_10_0_layers_1(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_10_0_layers_3(x)
        x = self.layers_10_0_layers_4(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_10_0_layers_6(x)
        x = self.layers_10_0_layers_7(x)
        x_main = x
        x_main = self.layers_10_1_layers_0(x_main)
        x_main = self.layers_10_1_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_10_1_layers_3(x_main)
        x_main = self.layers_10_1_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_10_1_layers_6(x_main)
        x_main = self.layers_10_1_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_10_2_layers_0(x_main)
        x_main = self.layers_10_2_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_10_2_layers_3(x_main)
        x_main = self.layers_10_2_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_10_2_layers_6(x_main)
        x_main = self.layers_10_2_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.layers_11_0_layers_0(x)
        x = self.layers_11_0_layers_1(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_11_0_layers_3(x)
        x = self.layers_11_0_layers_4(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_11_0_layers_6(x)
        x = self.layers_11_0_layers_7(x)
        x_main = x
        x_main = self.layers_11_1_layers_0(x_main)
        x_main = self.layers_11_1_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_11_1_layers_3(x_main)
        x_main = self.layers_11_1_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_11_1_layers_6(x_main)
        x_main = self.layers_11_1_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.layers_12_0_layers_0(x)
        x = self.layers_12_0_layers_1(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_12_0_layers_3(x)
        x = self.layers_12_0_layers_4(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_12_0_layers_6(x)
        x = self.layers_12_0_layers_7(x)
        x_main = x
        x_main = self.layers_12_1_layers_0(x_main)
        x_main = self.layers_12_1_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_12_1_layers_3(x_main)
        x_main = self.layers_12_1_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_12_1_layers_6(x_main)
        x_main = self.layers_12_1_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_12_2_layers_0(x_main)
        x_main = self.layers_12_2_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_12_2_layers_3(x_main)
        x_main = self.layers_12_2_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_12_2_layers_6(x_main)
        x_main = self.layers_12_2_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_12_3_layers_0(x_main)
        x_main = self.layers_12_3_layers_1(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_12_3_layers_3(x_main)
        x_main = self.layers_12_3_layers_4(x_main)
        x_main = F.relu6(x_main, inplace=True)
        x_main = self.layers_12_3_layers_6(x_main)
        x_main = self.layers_12_3_layers_7(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.layers_13_0_layers_0(x)
        x = self.layers_13_0_layers_1(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_13_0_layers_3(x)
        x = self.layers_13_0_layers_4(x)
        x = F.relu6(x, inplace=True)
        x = self.layers_13_0_layers_6(x)
        x = self.layers_13_0_layers_7(x)
        x = self.layers_14(x)
        x = self.layers_15(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_1(x)
        return x

def mnasnet(**kwargs):
    return MnasNet(**kwargs)
