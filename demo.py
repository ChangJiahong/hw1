
####################
# 手写梯度下降 与 线性回归模型
####################

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import linear_model

# x_data = [[338.], [333.], [328.], [207.], [226.], [25.], [179.], [60.], [208.], [606.]]
x_data = [[338.], [333.], [328.], [207.], [226.], [25.], [179.], [60.], [208.], [606.]]

y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

# ydata = b + w * xdata


x = np.arange(-200, -100, 1)  # x 坐标轴的值
y = np.arange(-5, 5, 0.1)    # y 坐标轴的值
Z = np.zeros((len(x), len(y)))  # z 是 矩阵0
X, Y = np.meshgrid(x, y)  # 网格点坐标


for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n][0])**2
        Z[j][i] = Z[j][i] / len(x_data)


b = -120
w = -4
lr = 0.0000001
iteration = 100000

b_his = [b]
w_his = [w]

for i in range(iteration):

    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n][0]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n][0]) * x_data[n][0]

    b = b - lr * b_grad
    w = w - lr * w_grad

    b_his.append(b)
    w_his.append(w)

reg = linear_model.LinearRegression()
print(reg.fit(x_data, y_data))
print(reg.coef_[0])
print(reg.intercept_)


plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
# plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_his, w_his, 'o-', ms=3, lw=1.5, color='black')
plt.plot(reg.intercept_, reg.coef_[0], 'x', ms=12, lw=1.5, color='orange')

plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()