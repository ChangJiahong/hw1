# -*- coding: utf-8 -*-
# @File    : demo2.py
# @Author  : CJH
# @Date    : 2019/4/9
# @Software: PyCharm
# @Desc    : 天气PM2.5预测


import csv
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import linear_model

trainingData = 'trainingData/'

x_data = []

y_data = []

with open(trainingData+'xData_pm25.csv', encoding='utf-8') as f:
    x_data = np.loadtxt(f, delimiter=",")

with open(trainingData+'yData_pm25.csv') as f:
    y_data = np.loadtxt(f, delimiter=",")

# regs = linear_model.RidgeCV(np.linspace(1,1000))
# regs.fit(x_data, y_data)
#
# alpha = regs.alpha_


reg = linear_model.Ridge(alpha=898.21)
reg.fit(x_data, y_data)

# 14,16,13,14,21,19,23,18,17
# 26,39,36,35,31,28,25,20,19
# 33,39,39,25,18,18,17,9,4
# 50,70,70,73,61,55,47,38,26

test_x_data = []

test_y_data = []

testingData = 'testingData/'

with open(testingData+'xData_pm25.csv', encoding='utf-8') as f:
    test_x_data = np.loadtxt(f, delimiter=",")

with open(testingData+'yData_pm25.csv') as f:
    test_y_data = np.loadtxt(f, delimiter=",")

pre_y_data = reg.predict(test_x_data)

err = abs(pre_y_data - test_y_data)

# plt.plot(np.linspace(1, len(err), len(err)), err)
# plt.xlabel(r'$trainingIndex$', fontsize=16)
# plt.ylabel(r'$error$', fontsize=16)
# plt.title('abs trainingError')
# plt.show()

print(err.sum(axis=0)/len(err)*1.0)
