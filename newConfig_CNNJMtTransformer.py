"""
This is a standard script for the FBtCNNDing2021
"""
# @网络参数
num_fb = 4
C = 9
F = 250
drop_rate = 0.5
N = 4

# @数据预处理参数
tw = 1
down_sample = 4  # 下采样设置
fs = 1000 / down_sample  # fs为float类型
win_data = int(fs * tw)

f_down1 = 3  # 第一个滤波器
f_up1 = 14

f_down2 = 9  # 第二个滤波器
f_up2 = 26

f_down3 = 14  # 第三个滤波器
f_up3 = 38

f_down4 = 19  # 第四个滤波器
f_up4 = 50

# mask_rate = 0.1  # 掩码比例

# @训练参数
num_sub = 2
num_data = 5120  # 512 1024 5120 10240

bth_size = 128
# num_epo = 700
num_epo = 2
channel = 9  # 选取的通道数
lr = 0.7e-3



