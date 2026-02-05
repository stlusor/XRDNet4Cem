import torch
import torch.nn as nn
import torch.nn.functional as F


# 注意：打包推理时不需要 sklearn，这里删除了 import numpy 和 sklearn
# 如果你在 forward 里用到了 numpy，请取消下面这一行的注释
# import numpy as np

# ==========================================
# 辅助模块定义
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TimeFrequencyAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(TimeFrequencyAttention, self).__init__()
        self.freq_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.time_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        freq_weight = self.freq_att(x)
        time_weight = self.time_att(x)
        return x * freq_weight * time_weight


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        reduced_channels = in_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_channels * 4, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1, branch3, branch5, branch_pool]
        fusion = torch.cat(outputs, dim=1)
        output = self.fusion(fusion)
        return output + x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_cat))
        return x * attention_map


# ==========================================
# 主模型定义
# ==========================================
class XRD_CNN_CWT(nn.Module):
    def __init__(self):
        super(XRD_CNN_CWT, self).__init__()

        # 主干网络
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.se1 = SEBlock(16, reduction=4)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.se2 = SEBlock(32, reduction=8)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.se3 = SEBlock(64, reduction=8)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.msff_early = MultiScaleFeatureFusion(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)
        self.se4 = SEBlock(128, reduction=16)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(128)
        self.se5 = SEBlock(128, reduction=16)

        # 残差块
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)

        # 多尺度特征融合
        self.msff = MultiScaleFeatureFusion(128)

        # 时频注意力
        self.tf_attention = TimeFrequencyAttention(128, reduction=8)
        # 空间注意力
        self.spatial_attention = SpatialAttention(kernel_size=3)

        # ============ 中间层特征提取 ============
        self.mid_pool = nn.AdaptiveAvgPool2d(1)

        # ============ 全连接层 ============
        self.fc1 = None
        self.ln1 = None

        # 输出层固定为130
        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 256)
        self.ln4 = nn.LayerNorm(256)
        self.fc5 = nn.Linear(256, 130)  # 硬编码 130

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.2)

        self.fc_initialized = False

    def forward(self, x):
        # ============ 编码器：卷积层 ============
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = self.pool3(x)
        x = self.msff_early(x)

        mid_feat = self.mid_pool(x).view(x.size(0), -1)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
        x = self.pool4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.se5(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.msff(x)
        x = self.tf_attention(x)
        x = self.spatial_attention(x)

        x = x.view(x.size(0), -1)
        x = torch.cat([x, mid_feat], dim=1)

        if not self.fc_initialized:
            fc_input_size = x.shape[1]
            self.fc1 = nn.Linear(fc_input_size, 2048).to(x.device)
            self.ln1 = nn.LayerNorm(2048).to(x.device)
            # 在GUI运行时，通常不希望在后台打印太多日志，除非调试
            # print(f"Dynamic Init: {fc_input_size}")
            self.fc_initialized = True

        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.ln4(self.fc4(x)))
        x = self.fc5(x)

        return x