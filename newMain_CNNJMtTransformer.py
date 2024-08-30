import torch
from newMethod_JMtTransformerPlus import CNNJMtTransformer
import newConfig_CNNJMtTransformer as config
from torch.utils.data import DataLoader
from torch import nn, optim
from random import sample
import numpy as np
from newDP_CNNJMtTransformer import *
from scipy.io import savemat


def tra_val_tes_single_sub(net, optimizer, loss_f, loader_train, loader_val, loader_tes, sub, num_epo, num_data,
                           bth_size):
    # 单受试者日志（保存单个受试者数据-放在EPO循环外面）
    global tra_loss_sub_log, tra_pdt_sub_log, tra_tgt_sub_log
    global val_loss_sub_log, val_pdt_sub_log, val_tgt_sub_log, val_acc_sub_log
    global tes_loss_sub_log, tes_pdt_sub_log, tes_tgt_sub_log, tes_acc_sub_log

    # epo循环
    for epo in range(num_epo):
        # 单世代者日志（保存世代受试者数据-放在EPO循环里面，且BTH循环外面）
        tra_loss_epo_log, tra_pdt_epo_log, tra_tgt_epo_log = [], [], []
        val_loss_epo_log, val_pdt_epo_log, val_tgt_epo_log, val_acc_epo_log = [], [], [], []
        tes_loss_epo_log, tes_pdt_epo_log, tes_tgt_epo_log, tes_acc_epo_log = [], [], [], []

        # 单个世代循环的次数-(总数居量/batch_size)
        epo_size = int(num_data / bth_size)

        # @1网络训练
        net.train()
        # print('Start train:')
        for iter_idx_tra, bth_tra in enumerate(loader_train):
            if iter_idx_tra >= epo_size:  # 单世代循环训练退出条件
                break

            # get bth data
            data_tra_bth, tgt_tra_bth = bth_tra[0], bth_tra[1]
            data_tra_bth, tgt_tra_bth = data_tra_bth.to(device), tgt_tra_bth.to(device)

            # calculate loss
            optimizer.zero_grad()  # 优化器梯度清零
            output_tra_bth = net(data_tra_bth)
            loss_tra_bth = loss_f(output_tra_bth, tgt_tra_bth.long())
            loss_tra_bth.backward()
            optimizer.step()

            # 剥离数据
            tmp_pdt_tra_bth = np.argmax(output_tra_bth.data.cpu().numpy(), axis=1)
            tmp_tgt_tra_bth = tgt_tra_bth.data.cpu().numpy()  # 转化成numpy

            # 保存BTH数据
            tra_loss_epo_log.append(loss_tra_bth.data.cpu().numpy())
            tra_pdt_epo_log.extend(list(tmp_pdt_tra_bth))
            tra_tgt_epo_log.extend(list(tmp_tgt_tra_bth))

        # 保存EPO数据
        tra_loss_sub_log.append([epo, np.mean(tra_loss_epo_log)])
        tra_pdt_sub_log.append([epo, tra_pdt_epo_log])
        tra_tgt_sub_log.append([epo, tra_tgt_epo_log])
        # print('End train!')

        # @网络验证
        net.eval()
        # print('Start validation:')
        for iter_idx_val, bth_val in enumerate(loader_val):
            if iter_idx_val >= epo_size:  # 单世代循环训练退出条件
                break

            # get bth data
            data_val_bth, tgt_val_bth = bth_val[0], bth_val[1]
            data_val_bth, tgt_val_bth = data_val_bth.to(device), tgt_val_bth.to(device)

            # calculate loss
            with torch.no_grad():  # 取消梯度计算环节
                optimizer.zero_grad()
                output_val_bth = net(data_val_bth)
                loss_val_bth = loss_f(output_val_bth, tgt_val_bth.long())

                # 剥离数据
                tmp_pdt_val_bth = np.argmax(output_val_bth.data.cpu().numpy(), axis=1)
                tmp_tgt_val_bth = tgt_val_bth.data.cpu().numpy()  # 转化成numpy

                # 保存BTH数据
                val_loss_epo_log.append(loss_val_bth.data.cpu().numpy())
                val_pdt_epo_log.extend(list(tmp_pdt_val_bth))
                val_tgt_epo_log.extend(list(tmp_tgt_val_bth))

                # 计算BTH的ACC
                num_right = 0
                for i in range(bth_size):
                    if tmp_pdt_val_bth[i] == tmp_tgt_val_bth[i]:
                        num_right += 1
                val_bth_acc = num_right / bth_size
                val_acc_epo_log.append(val_bth_acc)

        # 保存EPO数据
        val_loss_sub_log.append([epo, np.mean(val_loss_epo_log)])
        val_pdt_sub_log.append([epo, val_pdt_epo_log])
        val_tgt_sub_log.append([epo, val_tgt_epo_log])
        val_acc_sub_log.append([epo, np.mean(val_acc_epo_log)])
        # print('Finish validation!')

        # @网络测试
        net.eval()
        # print('Start test:')
        for iter_idx_tes, bth_tes in enumerate(loader_tes):
            if iter_idx_tes >= epo_size:  # 单世代循环训练退出条件
                break

            # get bth data
            data_tes_bth, tgt_tes_bth = bth_tes[0], bth_tes[1]
            data_tes_bth, tgt_tes_bth = data_tes_bth.to(device), tgt_tes_bth.to(device)

            # calculate loss
            with torch.no_grad():  # 取消梯度计算环节
                optimizer.zero_grad()
                output_tes_bth = net(data_tes_bth)
                loss_tes_bth = loss_f(output_tes_bth, tgt_tes_bth.long())

                # 剥离数据
                tmp_pdt_tes_bth = np.argmax(output_tes_bth.data.cpu().numpy(), axis=1)
                tmp_tgt_tes_bth = tgt_tes_bth.data.cpu().numpy()  # 转化成numpy

                # 保存BTH数据
                tes_loss_epo_log.append(loss_tes_bth.data.cpu().numpy())
                tes_pdt_epo_log.extend(list(tmp_pdt_tes_bth))
                tes_tgt_epo_log.extend(list(tmp_tgt_tes_bth))

                # 计算BTH的ACC
                num_right = 0
                for i in range(bth_size):
                    if tmp_pdt_tes_bth[i] == tmp_tgt_tes_bth[i]:
                        num_right += 1
                tes_bth_acc = num_right / bth_size
                tes_acc_epo_log.append(tes_bth_acc)

        # 保存EPO数据
        tes_loss_sub_log.append([epo, np.mean(tes_loss_epo_log)])
        tes_pdt_sub_log.append([epo, tes_pdt_epo_log])
        tes_tgt_sub_log.append([epo, tes_tgt_epo_log])
        tes_acc_sub_log.append([epo, np.mean(tes_acc_epo_log)])
        # print('Finish test!')

        # @单世代信息反馈
        print('Sub %d, epo %d, loss_tra %.3f, loss_val is %.3f, loss_tes is %.3f, acc_val %.3f, acc_tes %.3f'
              % (sub, epo, float(np.mean(tra_loss_epo_log)), float(np.mean(val_loss_epo_log)),
                 float(np.mean(tes_loss_epo_log)), float(np.mean(val_acc_epo_log)),
                 float(np.mean(tes_acc_epo_log))))

    path_tra_loss_sub_log = './stdLog/Sub%d_tra_loss_sub_log.mat' % sub
    path_val_loss_sub_log = './stdLog/Sub%d_val_loss_sub_log.mat' % sub
    path_tes_loss_sub_log = './stdLog/Sub%d_tes_loss_sub_log.mat' % sub
    path_val_acc_sub_log = './stdLog/Sub%d_val_acc_sub_log.mat' % sub
    path_tes_acc_sub_log = './stdLog/Sub%d_tes_acc_sub_log.mat' % sub
    path_tes_pdt_sub_log = './stdLog/Sub%d_tes_pdt_sub_log.mat' % sub
    path_tes_tgt_sub_log = './stdLog/Sub%d_tes_tgt_sub_log.mat' % sub

    savemat(path_tra_loss_sub_log, {'tra_loss_sub_log': tra_loss_sub_log})
    savemat(path_val_loss_sub_log, {'val_loss_sub_log': val_loss_sub_log})
    savemat(path_tes_loss_sub_log, {'tes_loss_sub_log': tes_loss_sub_log})
    savemat(path_val_acc_sub_log, {'val_acc_sub_log': val_acc_sub_log})
    savemat(path_tes_acc_sub_log, {'tes_acc_sub_log': tes_acc_sub_log})
    savemat(path_tes_pdt_sub_log, {'tes_pdt_sub_log': tes_pdt_sub_log})
    savemat(path_tes_tgt_sub_log, {'tes_tgt_sub_log': tes_tgt_sub_log})

    return None


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is', device)

    # @数据预处理
    for sub in range(1, config.num_sub + 1):
        if sub < 10:
            path_sub = "./stdData/sess01_FBtCNNDing2021/sess01_subj0%d_EEG_SSVEP.mat" % sub
        else:
            path_sub = "./stdData/sess01_FBtCNNDing2021/sess01_subj%d_EEG_SSVEP.mat" % sub

        # 对100次实验随机划分10个用于验证，其余用于训练-(例如，sess01数据集有100组训练实验和测试实验)
        list_tra_val = list(range(100))
        list_val = sample(list_tra_val, 10)
        list_tra = [list_tra_val[i] for i in range(len(list_tra_val)) if (i not in list_val)]
        win_data = int(config.fs * config.tw)  # 时间窗口对应帧数

        # 获取滤波后的训练-验证数据、标签和起始时间-(数据预处理：滤波与通道选择)
        data_tra_tmp1, data_tra_tmp2, data_tra_tmp3, data_tra_tmp4, label_tra, start_time_tra \
            = get_train_data(config.f_down1, config.f_up1, config.f_down2, config.f_up2, config.f_down3, config.f_up3,
                             config.f_down4, config.f_up4, path_sub, config.down_sample, config.fs)

        data_tra = [data_tra_tmp1, data_tra_tmp2, data_tra_tmp3, data_tra_tmp4]  # 数据聚合, 形状变为4*9*(all)-4个波段
        data_tra = np.array(data_tra)

        # 获取滤波后的测试数据、标签和起始时间
        data_tes_tmp1, data_tes_tmp2, data_tes_tmp3, data_tes_tmp4, label_tes, start_time_tes \
            = get_test_data(config.f_down1, config.f_up1, config.f_down2, config.f_up2, config.f_down3, config.f_up3,
                            config.f_down4, config.f_up4, path_sub, config.down_sample, config.fs)

        data_tes = [data_tes_tmp1, data_tes_tmp2, data_tes_tmp3, data_tes_tmp4]
        data_tes = np.array(data_tes)

        # 数据集及加载器
        dataset_tra = train_BCIDataset_FBtCNN(config.num_data, data_tra, win_data, label_tra, start_time_tra,
                                              config.down_sample, list_tra, config.channel)
        loader_tra = DataLoader(dataset_tra, shuffle=True, batch_size=config.bth_size, num_workers=1,
                                pin_memory=True, drop_last=True)

        dataset_val = val_BCIDataset_FBtCNN(config.num_data, data_tra, win_data, label_tra, start_time_tra,
                                            config.down_sample, list_val, config.channel)
        loader_val = DataLoader(dataset_val, shuffle=True, batch_size=config.bth_size, num_workers=1,
                                pin_memory=True, drop_last=True)

        dataset_tes = test_BCIDataset_FBtCNN(config.num_data, data_tes, win_data, label_tes, start_time_tes,
                                             config.down_sample, config.channel)
        loader_tes = DataLoader(dataset_tes, shuffle=True, batch_size=config.bth_size, num_workers=1,
                                pin_memory=True, drop_last=True)

        # @网络-损失-优化器设计
        # net = JTransformer(config.C, config.F, config.N)
        net = CNNJMtTransformer(config.num_fb, config.C, config.F, config.drop_rate, config.N)
        net.to(device)  # Here we are !
        loss_f = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), config.lr, weight_decay=0.01)  # 对参数进行正则化weight_decay, 防止过拟合

        # @网络训练-验证-测试
        tra_loss_sub_log, tra_pdt_sub_log, tra_tgt_sub_log = [], [], []
        val_loss_sub_log, val_pdt_sub_log, val_tgt_sub_log, val_acc_sub_log = [], [], [], []
        tes_loss_sub_log, tes_pdt_sub_log, tes_tgt_sub_log, tes_acc_sub_log = [], [], [], []

        tra_val_tes_single_sub(net, optimizer, loss_f, loader_tra, loader_val, loader_tes, sub, config.num_epo,
                               config.num_data, config.bth_size)

    print('The Main Function Is Over!')
