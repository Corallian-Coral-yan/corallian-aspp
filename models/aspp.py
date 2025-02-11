import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        # 1x1 Convolution
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Atrous Convolutions with different dilation rates
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=atrous_rates[0], dilation=atrous_rates[0], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=atrous_rates[1], dilation=atrous_rates[1], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=atrous_rates[2], dilation=atrous_rates[2], bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_g = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        # Final 1x1 Convolution to fuse all outputs
        self.conv1x1_out = nn.Conv2d(
            out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1x1 = self.relu(self.bn1(self.conv1x1(x)))
        conv3x3_1 = self.relu(self.bn2(self.conv3x3_1(x)))
        conv3x3_2 = self.relu(self.bn3(self.conv3x3_2(x)))
        conv3x3_3 = self.relu(self.bn4(self.conv3x3_3(x)))

        # Global Average Pooling branch
        global_avg = self.global_avg_pool(x)
        global_avg = self.relu(self.bn5(self.conv1x1_g(global_avg)))
        global_avg = F.interpolate(
            global_avg, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate all branches
        output = torch.cat([conv1x1, conv3x3_1, conv3x3_2,
                           conv3x3_3, global_avg], dim=1)
        output = self.relu(self.bn_out(self.conv1x1_out(output)))

        return output
