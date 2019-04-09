# -*- coding: utf-8 -*-
# @File    : xyData.py
# @Author  : CJH
# @Date    : 2019/4/9
# @Software: PyCharm
# @Desc    : 数据处理

import csv
from numpy import *
import numpy as np
from tqdm import tqdm

# 特征值 pm2.5 前8个小时的pm2.5的值
train_x_data = []

# 第9个小时的pm2.5值
train_y_data = []

filename = 'trainingData/train.csv'

with open(filename, encoding='utf-8') as f:
    reader = csv.reader(f)
    ls = list(reader)

    k = 0
    for i in tqdm(range(10, len(ls), 18)):
        k += 1
        # print(str(k) + ' ' + ls[i][2])

        for j in range(3, 19):
            # print(ls[i][j:j+8])
            train_x_data.append(ls[i][j:j+8])
            # print(ls[i][j+8])
            train_y_data.append(ls[i][j+8])


train_x_data = np.array(train_x_data, float)
train_y_data = np.array(train_y_data, float)

np.savetxt("trainingData/xData_pm25.csv", train_x_data, fmt='%.1f', delimiter=",")
np.savetxt("trainingData/yData_pm25.csv", train_y_data, fmt='%.1f', delimiter=",")


testData = 'testingData/'

test_x_data = []
test_y_data = []
with open(testData+'test.csv', encoding="GBK") as f:
    reader = csv.reader(f)
    ls = list(reader)
    k = 0
    for i in tqdm(range(9, len(ls), 18)):
        k += 1
        test_x_data.append(ls[i][2:10])
        test_y_data.append(ls[i][10])

test_x_data = np.array(test_x_data, float)
test_y_data = np.array(test_y_data, float)

print(test_x_data.shape)
print(test_y_data.shape)

np.savetxt(testData+"xData_pm25.csv", test_x_data, fmt='%.1f', delimiter=",")
np.savetxt(testData+"yData_pm25.csv", test_y_data, fmt='%.1f', delimiter=",")


