from Preprecess import preprocess
import numpy as np
import os
import random
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from functools import partial
import math
import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
# from torch._six import container_abcs
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import time
# from sageattention import sageattn
# import torch.nn.functional as F
# F.scaled_dot_product_attention = sageattn
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE







dataset = 'Bench'
totalsubject = 2
totalblock = 6
signal_length=0.2
totalcharacter=40
sampling_rate=250
visual_latency=0.14
visual_cue = 0.5
sample_length = 50
total_ch = 64
total_delay=visual_latency+visual_cue
delay_sample_point=round(total_delay*sampling_rate)
sample_interval = np.arange(delay_sample_point + 1, delay_sample_point + sample_length + 1)
channels = [48,54,55,56,57,58,61,62,63];
subban_no = 3



AllData, y_AllData = preprocess(channels, sample_length, sample_interval, subban_no, totalsubject, totalblock, totalcharacter, sampling_rate, dataset)
data=AllData
y_data=y_AllData
testaccuracy = np.zeros((totalsubject, totalblock))
for s in range(totalsubject):
    for block in range(totalblock):
        allblock = list(range(totalblock))
        allblock.pop(block)

        # 获取训练数据并调整维度
        train1 = data[:, :, :, :, allblock, s]  # 维度为 (9, 50, 3, 40, 5)
        train2 = train1.permute(4, 3, 2, 0, 1)  # 转置为 (5, 40, 3, 9, 50)#permute,transpose
        train = train2.reshape(-1, train2.shape[2],train2.shape[3], train2.shape[4])  # 重新调整为 (200, 9, 50)

        # 获取训练标签并调整维度
        train_y1 = y_data[:, :, allblock, s]  # 维度为 (1, 40, 5)
        train_y2 = train_y1.transpose(2, 1, 0)  # 转置为 (5, 40, 1)
        train_y = train_y2.reshape(-1)-1  # 重新调整为 (200,)

        # 处理测试数据
        test1 = data[:, :, :, :, block, s]  # 维度为 (9, 50, 3, 40)
        test = test1.permute(3, 2, 0, 1)  # 重新调整为 (40, 9, 50)
        # 处理测试标签
        test_y1 = y_data[:, :, block, s]  # 维度为 (1, 40)
        test_y = test_y1.squeeze() - 1  # 去掉多余维度并调整标签

        # 将数据转换为 PyTorch 张量
        train = torch.tensor(train, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.long)
        test = torch.tensor(test, dtype=torch.float32)
        test_y = torch.tensor(test_y, dtype=torch.long)
