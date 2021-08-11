import torch
import torch.nn as nn
import torch.nn.functional as F

class MnasNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layers_0 = nn.Conv2d(3, 26, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layers_1 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_3 = nn.Conv2d(26, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=26, bias=False)
        self.layers_4 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_6 = nn.Conv2d(26, 11, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_7 = nn.BatchNorm2d(11, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_0 = nn.Conv2d(11, 44, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_1 = nn.BatchNorm2d(44, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_3 = nn.Conv2d(44, 44, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=44, bias=False)
        self.layers_8_0_layers_4 = nn.BatchNorm2d(44, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_6 = nn.Conv2d(44, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_7 = nn.BatchNorm2d(17, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_0 = nn.Conv2d(17, 54, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_1 = nn.BatchNorm2d(54, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_3 = nn.Conv2d(54, 54, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=54, bias=False)
        self.layers_8_1_layers_4 = nn.BatchNorm2d(54, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_6 = nn.Conv2d(54, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_7 = nn.BatchNorm2d(17, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_0 = nn.Conv2d(17, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_2_layers_1 = nn.BatchNorm2d(56, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_3 = nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=56, bias=False)
        self.layers_8_2_layers_4 = nn.BatchNorm2d(56, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_6 = nn.Conv2d(56, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_2_layers_7 = nn.BatchNorm2d(17, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_0 = nn.Conv2d(17, 57, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_1 = nn.BatchNorm2d(57, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_3 = nn.Conv2d(57, 57, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=57, bias=False)
        self.layers_9_0_layers_4 = nn.BatchNorm2d(57, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_6 = nn.Conv2d(57, 30, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_7 = nn.BatchNorm2d(30, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_0 = nn.Conv2d(30, 98, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_1 = nn.BatchNorm2d(98, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_3 = nn.Conv2d(98, 98, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=98, bias=False)
        self.layers_9_1_layers_4 = nn.BatchNorm2d(98, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_6 = nn.Conv2d(98, 30, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_7 = nn.BatchNorm2d(30, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_0 = nn.Conv2d(30, 91, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_1 = nn.BatchNorm2d(91, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_3 = nn.Conv2d(91, 91, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=91, bias=False)
        self.layers_9_2_layers_4 = nn.BatchNorm2d(91, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_6 = nn.Conv2d(91, 30, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_7 = nn.BatchNorm2d(30, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_0 = nn.Conv2d(30, 190, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_1 = nn.BatchNorm2d(190, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_3 = nn.Conv2d(190, 190, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=190, bias=False)
        self.layers_10_0_layers_4 = nn.BatchNorm2d(190, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_6 = nn.Conv2d(190, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_7 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_0 = nn.Conv2d(64, 391, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_1 = nn.BatchNorm2d(391, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_3 = nn.Conv2d(391, 391, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=391, bias=False)
        self.layers_10_1_layers_4 = nn.BatchNorm2d(391, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_6 = nn.Conv2d(391, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_7 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_0 = nn.Conv2d(64, 379, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_1 = nn.BatchNorm2d(379, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_3 = nn.Conv2d(379, 379, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=379, bias=False)
        self.layers_10_2_layers_4 = nn.BatchNorm2d(379, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_6 = nn.Conv2d(379, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_7 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_0 = nn.Conv2d(64, 396, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_1 = nn.BatchNorm2d(396, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_3 = nn.Conv2d(396, 396, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=396, bias=False)
        self.layers_11_0_layers_4 = nn.BatchNorm2d(396, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_6 = nn.Conv2d(396, 76, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_7 = nn.BatchNorm2d(76, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_0 = nn.Conv2d(76, 471, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_1 = nn.BatchNorm2d(471, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_3 = nn.Conv2d(471, 471, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=471, bias=False)
        self.layers_11_1_layers_4 = nn.BatchNorm2d(471, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_6 = nn.Conv2d(471, 76, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_7 = nn.BatchNorm2d(76, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_0 = nn.Conv2d(76, 479, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_1 = nn.BatchNorm2d(479, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_3 = nn.Conv2d(479, 479, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=479, bias=False)
        self.layers_12_0_layers_4 = nn.BatchNorm2d(479, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_6 = nn.Conv2d(479, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_7 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_0 = nn.Conv2d(128, 931, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_1 = nn.BatchNorm2d(931, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_3 = nn.Conv2d(931, 931, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=931, bias=False)
        self.layers_12_1_layers_4 = nn.BatchNorm2d(931, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_6 = nn.Conv2d(931, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_7 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_0 = nn.Conv2d(128, 919, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_1 = nn.BatchNorm2d(919, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_3 = nn.Conv2d(919, 919, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=919, bias=False)
        self.layers_12_2_layers_4 = nn.BatchNorm2d(919, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_6 = nn.Conv2d(919, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_7 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_0 = nn.Conv2d(128, 922, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_1 = nn.BatchNorm2d(922, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_3 = nn.Conv2d(922, 922, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=922, bias=False)
        self.layers_12_3_layers_4 = nn.BatchNorm2d(922, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_6 = nn.Conv2d(922, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_7 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_0 = nn.Conv2d(128, 926, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_1 = nn.BatchNorm2d(926, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_3 = nn.Conv2d(926, 926, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=926, bias=False)
        self.layers_13_0_layers_4 = nn.BatchNorm2d(926, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_6 = nn.Conv2d(926, 291, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_7 = nn.BatchNorm2d(291, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_14 = nn.Conv2d(291, 1137, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_15 = nn.BatchNorm2d(1137, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier_1 = nn.Linear(in_features=1137, out_features=1000, bias=True)

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