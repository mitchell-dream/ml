"""
===========================================================
Plot Ridge coefficients as a function of the regularization
===========================================================

Shows the effect of collinearity in the coefficients of an estimator.

.. currentmodule:: sklearn.linear_model

:class:`Ridge` Regression is the estimator used in this example.
Each color represents a different feature of the
coefficient vector, and this is displayed as a function of the
regularization parameter.

This example also shows the usefulness of applying Ridge regression
to highly ill-conditioned matrices. For such matrices, a slight
change in the target variable can cause huge variances in the
calculated weights. In such cases, it is useful to set a certain
regularization (alpha) to reduce this variation (noise).

When alpha is very large, the regularization effect dominates the
squared loss function and the coefficients tend to zero.
At the end of the path, as alpha tends toward zero
and the solution tends towards the ordinary least squares, coefficients
exhibit big oscillations. In practise it is necessary to tune alpha
in such a way that a balance is maintained between both.
"""

# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# License: BSD 3 clause

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

### 本例展示 岭回归系数 作为正则化的函数的影响

### 岭回归为了解决线性回归中出现的过拟合以及在通过正规方程求解θ的过程中出现 x 转置乘以x 不可逆这两类问题的；
### 通过在损失函数中引入正则化来达到目的。

# 其中λ称为正则化参数，如果λ选取过大，会把所有参数θ均最小化，造成欠拟合，如果λ选取过小，会导致对过拟合问题解决不当，因此λ的选取是一个技术活。

# 总结
# 1.岭回归可以解决特征数量比样本量多的问题
# 2.岭回归作为一种缩减算法可以判断哪些特征重要或者不重要，有点类似于降维的效果
# 3.缩减算法可以看作是对一个模型增加偏差的同时减少方差
#
# 岭回归用于处理下面两类问题：
# 1.数据点少于变量个数
# 2.变量间存在共线性（最小二乘回归得到的系数不稳定，方差很大）


# X is the 10x10 Hilbert matrix
# X 是一个希尔伯特矩阵（对称矩阵，矩阵每个元素满足 Hij = 1/(i + j - 1)）
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# #############################################################################
# Compute paths

n_alphas = 200
# 创建 alpha 数组 开始时为 10^(-10) 结束为 10^(-2)，中间包含 n_alphas 项
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
# 查看岭回归 a 的大小 对估计出的参数的影响
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# #############################################################################
# Display results
# Get Current Axes 获得当前的Axes对象ax 然后可以用ax.plot()方法实现真正的绘图。
ax = plt.gca()

ax.plot(alphas, coefs)

#将x轴设置为log
ax.set_xscale('log')
# 设定 x 轴的范围
# plt.xlim(xmin=num1,xmax=num2)
# ax.set_xlim(-10,10) 设置 x 轴 初始值和末值
# ax.get_xlim() 返回 当前 x 的上下限
# 翻转 x 轴
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
