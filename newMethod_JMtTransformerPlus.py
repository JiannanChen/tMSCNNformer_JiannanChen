"""
This is a standard script for the SSVEPformerJBChen2023
"""
import torch
import torch.nn as nn
import newConfig_JMtTransformer as config


# ------------------------------------------------- JMtTransformer ----------------------------------------------------#
# @Channel combination
class blk_cha_comb(nn.Module):
    """
    目的：uses convolutional layers to perform weighted combination of multiple channels
    """

    def __init__(self, C, F):
        super(blk_cha_comb, self).__init__()
        self.cov1d = nn.Conv1d(C, 2 * C, 1, padding="same")
        self.ln = nn.LayerNorm([2 * C, F])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x):
        # 加入输入数据的形状是C*F, C for channel；利用形状Cx1或者1x1的卷积，对不同通道进行卷积，类似加权求和运算
        x = self.cov1d(x)
        x = self.ln(x)
        x = self.gelu(x)
        y = self.dropout(x)

        return y


# @SSVEPformer encoder
class blk_encoder(nn.Module):
    def __init__(self, C, F):
        super(blk_encoder, self).__init__()
        # para
        self.para_C = 2 * C
        self.C = C
        self.F = F

        # CNN module
        self.cnn_ln = nn.LayerNorm([2 * C, F])
        self.cnn_cov1d = nn.Conv1d(2 * C, 2 * C, 31, padding="same")
        self.cnn_ln2 = nn.LayerNorm([2 * C, F])
        self.cnn_gelu = nn.GELU()
        self.cnn_dropout = nn.Dropout(config.drop_rate)

        # Channel MLP module
        self.mlp_ln = nn.LayerNorm([2 * C, F])
        self.mlp_linear = nn.Linear(F, F)  # 这个Linear有问题？
        self.mlp_gelu = nn.GELU()
        self.mlp_dropout = nn.Dropout(config.drop_rate)

    def forward(self, x_0):
        # CNN module
        x = self.cnn_ln(x_0)
        x = self.cnn_cov1d(x)
        x = self.cnn_ln2(x)
        x = self.cnn_gelu(x)
        x = self.cnn_dropout(x)
        x_1 = x_0 + x

        # Channel MLP module
        x = self.mlp_ln(x_1)

        for idx in range(self.para_C):
            y = x[:, idx, :]
            y = self.mlp_linear(y)
            z = torch.unsqueeze(y, 1)
            if idx == 0:
                zz = z
            else:
                zz = torch.cat([zz, z], 1)

        x = self.mlp_gelu(zz)
        x = self.mlp_dropout(x)
        y = x_1 + x

        return y


# @MLP head
class blk_mlp(nn.Module):
    def __init__(self, C, F, N):
        super(blk_mlp, self).__init__()
        self.para_N = N
        self.dropout = nn.Dropout(config.drop_rate)
        self.linear = nn.Linear(2 * C * F, 6 * N)
        self.ln = nn.LayerNorm(6 * N)
        self.gelu = nn.GELU()
        self.dropout2 = nn.Dropout(config.drop_rate)
        self.linear2 = nn.Linear(6 * N, N)

    def forward(self, x):
        x = torch.flatten(x, 1, 2)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        out = self.linear2(x)

        return out


# @SSVEPformer
class JTransformer(nn.Module):
    def __init__(self, C, F, N):
        super(JTransformer, self).__init__()
        self.blk_cc = blk_cha_comb(C, F)
        self.blk_se_1 = blk_encoder(C, F)
        self.blk_se_2 = blk_encoder(C, F)
        self.blk_head = blk_mlp(C, F, N)

    def forward(self, x):
        x = self.blk_cc(x)
        x = self.blk_se_1(x)
        x = self.blk_se_2(x)
        y = self.blk_head(x)

        return y
# ---------------------------------------------------------------------------------------------------------------------#


# ------------------------------------------------ CNNJMtTransformer --------------------------------------------------#
class conv1D_block_(nn.Module):
    def __init__(self, in_channel, out_channel, k_size, stride, drop_rate):
        super(conv1D_block_, self).__init__()
        self.dropout_1 = nn.Dropout(drop_rate)
        self.cnn_cov1d = nn.Conv2d(in_channel, out_channel, k_size, stride, padding="same")
        # self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn1 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.dropout_1(x)
        x = self.cnn_cov1d(x)
        x = self.bn1(x)
        y = self.elu(x)

        return y


class multi_scale_1D(nn.Module):
    def __init__(self, inc_1, out_channel, first_k, firt_step, drop_out1):
        super(multi_scale_1D, self).__init__()
        self.conv1D_block_1 = conv1D_block_(inc_1, out_channel, first_k, firt_step, drop_out1)

        # self.conv1D_block_2 = conv1D_block_(out_channel, out_channel, 32, 1, drop_out1)
        # self.conv1D_block_3 = conv1D_block_(out_channel, out_channel, 16, 1, drop_out1)
        # self.conv1D_block_4 = conv1D_block_(out_channel, out_channel, 11, 1, drop_out1)

        self.conv1D_block_2 = conv1D_block_(out_channel, out_channel, (1, 32), 1, drop_out1)
        self.conv1D_block_3 = conv1D_block_(out_channel, out_channel, (1, 16), 1, drop_out1)
        self.conv1D_block_4 = conv1D_block_(out_channel, out_channel, (1, 11), 1, drop_out1)

    def forward(self, x):
        x1 = self.conv1D_block_1(x)
        x2 = self.conv1D_block_1(x)
        x3 = self.conv1D_block_1(x)
        x4 = self.conv1D_block_1(x)

        x2_2 = self.conv1D_block_2(x2)
        x3_2 = x3 + x2_2
        x3_3 = self.conv1D_block_3(x3_2)
        x4_3 = x4 + x3_3
        x4_4 = self.conv1D_block_4(x4_3)

        y = x1 + x2_2 + x3_3 + x4_4

        return y


class CNN_BLK(nn.Module):
    """
    这个CNN模块是参考文章：A Novel Data Augmentation Approach Using Mask Encoding for
    Deep Learning-Based Asynchronous SSVEP-BCI
    """
    def __init__(self, num_fb, C, F, drop_rate):
        super(CNN_BLK, self).__init__()
        self.cov2d1 = nn.Conv2d(num_fb, 8, (C, 1), 1)  # 卷后形状为8@1*250
        self.bn1_1 = nn.BatchNorm2d(8)
        self.elu_1 = nn.ELU()

        self.multi_scale_1D = multi_scale_1D(8, 32, 1, 1, drop_rate)

        # 自己添加的卷积层
        self.cov2d2 = nn.Conv2d(32, 2*C, (1, 1), 1)
        self.bn1_2 = nn.BatchNorm2d(18)
        self.elu_2 = nn.ELU()

    def forward(self, x):
        x = self.cov2d1(x)
        x = self.bn1_1(x)
        x = self.elu_1(x)

        x = self.multi_scale_1D(x)

        x = self.cov2d2(x)
        x = self.bn1_2(x)
        y = self.elu_2(x)

        return y


class CNNJMtTransformer(nn.Module):
    """
    注意：在正式使用的时候，CNN_BLK()模块的BN层需要修改成Batchnorm2D()
    """
    def __init__(self, num_fb, C, F, drop_rate, N):
        super(CNNJMtTransformer, self).__init__()
        self.cnn_blk = CNN_BLK(num_fb, C, F, drop_rate)
        self.blk_se_1 = blk_encoder(C, F)
        self.blk_se_2 = blk_encoder(C, F)
        self.blk_head = blk_mlp(C, F, N)

    def forward(self, x):
        x = self.cnn_blk(x)  # 输出形状为：bth@18@1*F
        x = torch.squeeze(x)
        x = self.blk_se_1(x)  # Here We Are!
        x = self.blk_se_2(x)
        y = self.blk_head(x)

        return y
# ---------------------------------------------------------------------------------------------------------------------#

# if __name__ == '__main__':
#     x = torch.rand(4, 9, 250)
#     f = CNN_BLK(4, 9, 250, 0.5)
#     y = f(x)

    # x = torch.rand(8, 1, 250)
    # f = multi_scale_1D(8, 32, 1, 1, 0.5)
    # y = f(x)

    # 创建一个形状为(256, 9, 250)的随机张量
    # x = torch.rand(256, 9, 250)
    # net = JTransformer(9, 250, 4)
    # y = net(x)

    # print("Wait!")
