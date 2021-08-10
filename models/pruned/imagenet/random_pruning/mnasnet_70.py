import torch
import torch.nn as nn
import torch.nn.functional as F

class MnasNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layers_0 = nn.Conv2d(3, 10, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layers_1 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_3 = nn.Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=10, bias=False)
        self.layers_4 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_6 = nn.Conv2d(10, 9, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_7 = nn.BatchNorm2d(9, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_0 = nn.Conv2d(9, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_1 = nn.BatchNorm2d(17, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_3 = nn.Conv2d(17, 17, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=17, bias=False)
        self.layers_8_0_layers_4 = nn.BatchNorm2d(17, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_6 = nn.Conv2d(17, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_7 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_0 = nn.Conv2d(8, 29, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_1 = nn.BatchNorm2d(29, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_3 = nn.Conv2d(29, 29, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=29, bias=False)
        self.layers_8_1_layers_4 = nn.BatchNorm2d(29, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_6 = nn.Conv2d(29, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_7 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_0 = nn.Conv2d(8, 27, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_2_layers_1 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_3 = nn.Conv2d(27, 27, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=27, bias=False)
        self.layers_8_2_layers_4 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_6 = nn.Conv2d(27, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_2_layers_7 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_0 = nn.Conv2d(8, 21, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_1 = nn.BatchNorm2d(21, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_3 = nn.Conv2d(21, 21, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=21, bias=False)
        self.layers_9_0_layers_4 = nn.BatchNorm2d(21, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_6 = nn.Conv2d(21, 13, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_7 = nn.BatchNorm2d(13, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_0 = nn.Conv2d(13, 47, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_1 = nn.BatchNorm2d(47, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_3 = nn.Conv2d(47, 47, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=47, bias=False)
        self.layers_9_1_layers_4 = nn.BatchNorm2d(47, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_6 = nn.Conv2d(47, 13, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_7 = nn.BatchNorm2d(13, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_0 = nn.Conv2d(13, 51, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_1 = nn.BatchNorm2d(51, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_3 = nn.Conv2d(51, 51, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=51, bias=False)
        self.layers_9_2_layers_4 = nn.BatchNorm2d(51, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_6 = nn.Conv2d(51, 13, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_7 = nn.BatchNorm2d(13, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_0 = nn.Conv2d(13, 89, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_1 = nn.BatchNorm2d(89, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_3 = nn.Conv2d(89, 89, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=89, bias=False)
        self.layers_10_0_layers_4 = nn.BatchNorm2d(89, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_6 = nn.Conv2d(89, 25, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_7 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_0 = nn.Conv2d(25, 185, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_1 = nn.BatchNorm2d(185, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_3 = nn.Conv2d(185, 185, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=185, bias=False)
        self.layers_10_1_layers_4 = nn.BatchNorm2d(185, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_6 = nn.Conv2d(185, 25, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_7 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_0 = nn.Conv2d(25, 197, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_1 = nn.BatchNorm2d(197, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_3 = nn.Conv2d(197, 197, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=197, bias=False)
        self.layers_10_2_layers_4 = nn.BatchNorm2d(197, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_6 = nn.Conv2d(197, 25, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_7 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_0 = nn.Conv2d(25, 194, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_1 = nn.BatchNorm2d(194, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_3 = nn.Conv2d(194, 194, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=194, bias=False)
        self.layers_11_0_layers_4 = nn.BatchNorm2d(194, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_6 = nn.Conv2d(194, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_7 = nn.BatchNorm2d(40, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_0 = nn.Conv2d(40, 218, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_1 = nn.BatchNorm2d(218, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_3 = nn.Conv2d(218, 218, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=218, bias=False)
        self.layers_11_1_layers_4 = nn.BatchNorm2d(218, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_6 = nn.Conv2d(218, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_7 = nn.BatchNorm2d(40, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_0 = nn.Conv2d(40, 230, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_1 = nn.BatchNorm2d(230, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_3 = nn.Conv2d(230, 230, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=230, bias=False)
        self.layers_12_0_layers_4 = nn.BatchNorm2d(230, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_6 = nn.Conv2d(230, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_7 = nn.BatchNorm2d(58, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_0 = nn.Conv2d(58, 473, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_1 = nn.BatchNorm2d(473, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_3 = nn.Conv2d(473, 473, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=473, bias=False)
        self.layers_12_1_layers_4 = nn.BatchNorm2d(473, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_6 = nn.Conv2d(473, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_7 = nn.BatchNorm2d(58, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_0 = nn.Conv2d(58, 474, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_1 = nn.BatchNorm2d(474, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_3 = nn.Conv2d(474, 474, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=474, bias=False)
        self.layers_12_2_layers_4 = nn.BatchNorm2d(474, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_6 = nn.Conv2d(474, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_7 = nn.BatchNorm2d(58, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_0 = nn.Conv2d(58, 453, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_1 = nn.BatchNorm2d(453, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_3 = nn.Conv2d(453, 453, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=453, bias=False)
        self.layers_12_3_layers_4 = nn.BatchNorm2d(453, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_6 = nn.Conv2d(453, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_7 = nn.BatchNorm2d(58, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_0 = nn.Conv2d(58, 446, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_1 = nn.BatchNorm2d(446, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_3 = nn.Conv2d(446, 446, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=446, bias=False)
        self.layers_13_0_layers_4 = nn.BatchNorm2d(446, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_6 = nn.Conv2d(446, 195, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_7 = nn.BatchNorm2d(195, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_14 = nn.Conv2d(195, 798, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_15 = nn.BatchNorm2d(798, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier_1 = nn.Linear(in_features=798, out_features=1000, bias=True)

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
