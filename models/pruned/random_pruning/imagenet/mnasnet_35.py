import torch
import torch.nn as nn
import torch.nn.functional as F

class MnasNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layers_0 = nn.Conv2d(3, 27, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layers_1 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_3 = nn.Conv2d(27, 27, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=27, bias=False)
        self.layers_4 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_6 = nn.Conv2d(27, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_7 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_0 = nn.Conv2d(15, 39, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_1 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_3 = nn.Conv2d(39, 39, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=39, bias=False)
        self.layers_8_0_layers_4 = nn.BatchNorm2d(39, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_6 = nn.Conv2d(39, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_7 = nn.BatchNorm2d(17, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_0 = nn.Conv2d(17, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_1 = nn.BatchNorm2d(60, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_3 = nn.Conv2d(60, 60, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=60, bias=False)
        self.layers_8_1_layers_4 = nn.BatchNorm2d(60, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_6 = nn.Conv2d(60, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)
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
        self.layers_9_0_layers_6 = nn.Conv2d(57, 29, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_7 = nn.BatchNorm2d(29, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_0 = nn.Conv2d(29, 91, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_1 = nn.BatchNorm2d(91, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_3 = nn.Conv2d(91, 91, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=91, bias=False)
        self.layers_9_1_layers_4 = nn.BatchNorm2d(91, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_6 = nn.Conv2d(91, 29, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_7 = nn.BatchNorm2d(29, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_0 = nn.Conv2d(29, 94, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_1 = nn.BatchNorm2d(94, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_3 = nn.Conv2d(94, 94, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=94, bias=False)
        self.layers_9_2_layers_4 = nn.BatchNorm2d(94, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_6 = nn.Conv2d(94, 29, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_7 = nn.BatchNorm2d(29, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_0 = nn.Conv2d(29, 181, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_1 = nn.BatchNorm2d(181, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_3 = nn.Conv2d(181, 181, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=181, bias=False)
        self.layers_10_0_layers_4 = nn.BatchNorm2d(181, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_6 = nn.Conv2d(181, 51, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_7 = nn.BatchNorm2d(51, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_0 = nn.Conv2d(51, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_1 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_3 = nn.Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
        self.layers_10_1_layers_4 = nn.BatchNorm2d(384, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_6 = nn.Conv2d(384, 51, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_7 = nn.BatchNorm2d(51, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_0 = nn.Conv2d(51, 378, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_1 = nn.BatchNorm2d(378, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_3 = nn.Conv2d(378, 378, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=378, bias=False)
        self.layers_10_2_layers_4 = nn.BatchNorm2d(378, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_6 = nn.Conv2d(378, 51, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_7 = nn.BatchNorm2d(51, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_0 = nn.Conv2d(51, 368, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_1 = nn.BatchNorm2d(368, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_3 = nn.Conv2d(368, 368, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=368, bias=False)
        self.layers_11_0_layers_4 = nn.BatchNorm2d(368, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_6 = nn.Conv2d(368, 69, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_7 = nn.BatchNorm2d(69, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_0 = nn.Conv2d(69, 436, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_1 = nn.BatchNorm2d(436, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_3 = nn.Conv2d(436, 436, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=436, bias=False)
        self.layers_11_1_layers_4 = nn.BatchNorm2d(436, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_6 = nn.Conv2d(436, 69, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_7 = nn.BatchNorm2d(69, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_0 = nn.Conv2d(69, 449, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_1 = nn.BatchNorm2d(449, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_3 = nn.Conv2d(449, 449, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=449, bias=False)
        self.layers_12_0_layers_4 = nn.BatchNorm2d(449, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_6 = nn.Conv2d(449, 124, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_7 = nn.BatchNorm2d(124, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_0 = nn.Conv2d(124, 899, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_1 = nn.BatchNorm2d(899, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_3 = nn.Conv2d(899, 899, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=899, bias=False)
        self.layers_12_1_layers_4 = nn.BatchNorm2d(899, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_6 = nn.Conv2d(899, 124, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_7 = nn.BatchNorm2d(124, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_0 = nn.Conv2d(124, 894, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_1 = nn.BatchNorm2d(894, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_3 = nn.Conv2d(894, 894, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=894, bias=False)
        self.layers_12_2_layers_4 = nn.BatchNorm2d(894, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_6 = nn.Conv2d(894, 124, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_7 = nn.BatchNorm2d(124, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_0 = nn.Conv2d(124, 874, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_1 = nn.BatchNorm2d(874, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_3 = nn.Conv2d(874, 874, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=874, bias=False)
        self.layers_12_3_layers_4 = nn.BatchNorm2d(874, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_6 = nn.Conv2d(874, 124, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_7 = nn.BatchNorm2d(124, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_0 = nn.Conv2d(124, 872, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_1 = nn.BatchNorm2d(872, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_3 = nn.Conv2d(872, 872, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=872, bias=False)
        self.layers_13_0_layers_4 = nn.BatchNorm2d(872, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_6 = nn.Conv2d(872, 269, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_7 = nn.BatchNorm2d(269, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_14 = nn.Conv2d(269, 1115, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_15 = nn.BatchNorm2d(1115, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier_1 = nn.Linear(in_features=1115, out_features=1000, bias=True)

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