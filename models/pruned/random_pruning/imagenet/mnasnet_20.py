import torch
import torch.nn as nn
import torch.nn.functional as F

class MnasNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layers_0 = nn.Conv2d(3, 29, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layers_1 = nn.BatchNorm2d(29, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_3 = nn.Conv2d(29, 29, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=29, bias=False)
        self.layers_4 = nn.BatchNorm2d(29, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_6 = nn.Conv2d(29, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_7 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_0 = nn.Conv2d(16, 43, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_1 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_3 = nn.Conv2d(43, 43, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=43, bias=False)
        self.layers_8_0_layers_4 = nn.BatchNorm2d(43, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_0_layers_6 = nn.Conv2d(43, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_0_layers_7 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_0 = nn.Conv2d(16, 63, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_1 = nn.BatchNorm2d(63, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_3 = nn.Conv2d(63, 63, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=63, bias=False)
        self.layers_8_1_layers_4 = nn.BatchNorm2d(63, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_1_layers_6 = nn.Conv2d(63, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_1_layers_7 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_0 = nn.Conv2d(16, 59, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_2_layers_1 = nn.BatchNorm2d(59, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_3 = nn.Conv2d(59, 59, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=59, bias=False)
        self.layers_8_2_layers_4 = nn.BatchNorm2d(59, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_8_2_layers_6 = nn.Conv2d(59, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_2_layers_7 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_0 = nn.Conv2d(16, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_1 = nn.BatchNorm2d(60, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_3 = nn.Conv2d(60, 60, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=60, bias=False)
        self.layers_9_0_layers_4 = nn.BatchNorm2d(60, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_0_layers_6 = nn.Conv2d(60, 35, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_0_layers_7 = nn.BatchNorm2d(35, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_0 = nn.Conv2d(35, 105, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_1 = nn.BatchNorm2d(105, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_3 = nn.Conv2d(105, 105, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=105, bias=False)
        self.layers_9_1_layers_4 = nn.BatchNorm2d(105, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_1_layers_6 = nn.Conv2d(105, 35, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_1_layers_7 = nn.BatchNorm2d(35, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_0 = nn.Conv2d(35, 103, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_1 = nn.BatchNorm2d(103, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_3 = nn.Conv2d(103, 103, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=103, bias=False)
        self.layers_9_2_layers_4 = nn.BatchNorm2d(103, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_9_2_layers_6 = nn.Conv2d(103, 35, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_2_layers_7 = nn.BatchNorm2d(35, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_0 = nn.Conv2d(35, 210, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_1 = nn.BatchNorm2d(210, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_3 = nn.Conv2d(210, 210, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=210, bias=False)
        self.layers_10_0_layers_4 = nn.BatchNorm2d(210, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_0_layers_6 = nn.Conv2d(210, 67, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_0_layers_7 = nn.BatchNorm2d(67, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_0 = nn.Conv2d(67, 413, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_1 = nn.BatchNorm2d(413, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_3 = nn.Conv2d(413, 413, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=413, bias=False)
        self.layers_10_1_layers_4 = nn.BatchNorm2d(413, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_1_layers_6 = nn.Conv2d(413, 67, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_1_layers_7 = nn.BatchNorm2d(67, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_0 = nn.Conv2d(67, 422, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_1 = nn.BatchNorm2d(422, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_3 = nn.Conv2d(422, 422, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=422, bias=False)
        self.layers_10_2_layers_4 = nn.BatchNorm2d(422, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_10_2_layers_6 = nn.Conv2d(422, 67, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_2_layers_7 = nn.BatchNorm2d(67, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_0 = nn.Conv2d(67, 421, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_1 = nn.BatchNorm2d(421, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_3 = nn.Conv2d(421, 421, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=421, bias=False)
        self.layers_11_0_layers_4 = nn.BatchNorm2d(421, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_0_layers_6 = nn.Conv2d(421, 85, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_0_layers_7 = nn.BatchNorm2d(85, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_0 = nn.Conv2d(85, 508, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_1 = nn.BatchNorm2d(508, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_3 = nn.Conv2d(508, 508, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=508, bias=False)
        self.layers_11_1_layers_4 = nn.BatchNorm2d(508, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_11_1_layers_6 = nn.Conv2d(508, 85, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_1_layers_7 = nn.BatchNorm2d(85, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_0 = nn.Conv2d(85, 495, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_1 = nn.BatchNorm2d(495, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_3 = nn.Conv2d(495, 495, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=495, bias=False)
        self.layers_12_0_layers_4 = nn.BatchNorm2d(495, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_0_layers_6 = nn.Conv2d(495, 148, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_0_layers_7 = nn.BatchNorm2d(148, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_0 = nn.Conv2d(148, 1016, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_1 = nn.BatchNorm2d(1016, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_3 = nn.Conv2d(1016, 1016, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1016, bias=False)
        self.layers_12_1_layers_4 = nn.BatchNorm2d(1016, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_1_layers_6 = nn.Conv2d(1016, 148, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_1_layers_7 = nn.BatchNorm2d(148, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_0 = nn.Conv2d(148, 1007, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_1 = nn.BatchNorm2d(1007, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_3 = nn.Conv2d(1007, 1007, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1007, bias=False)
        self.layers_12_2_layers_4 = nn.BatchNorm2d(1007, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_2_layers_6 = nn.Conv2d(1007, 148, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_2_layers_7 = nn.BatchNorm2d(148, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_0 = nn.Conv2d(148, 1031, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_1 = nn.BatchNorm2d(1031, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_3 = nn.Conv2d(1031, 1031, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1031, bias=False)
        self.layers_12_3_layers_4 = nn.BatchNorm2d(1031, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_12_3_layers_6 = nn.Conv2d(1031, 148, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_3_layers_7 = nn.BatchNorm2d(148, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_0 = nn.Conv2d(148, 1003, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_1 = nn.BatchNorm2d(1003, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_3 = nn.Conv2d(1003, 1003, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1003, bias=False)
        self.layers_13_0_layers_4 = nn.BatchNorm2d(1003, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_13_0_layers_6 = nn.Conv2d(1003, 302, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_0_layers_7 = nn.BatchNorm2d(302, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.layers_14 = nn.Conv2d(302, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_15 = nn.BatchNorm2d(1200, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier_1 = nn.Linear(in_features=1200, out_features=1000, bias=True)

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
