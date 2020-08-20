#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the coefficient
of determination are also calculated.

"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
# 读取数据
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 打印一下 特征向量的 维度
print(diabetes_X.shape)
# Use only one feature
# 只使用一个特征向量
print (diabetes_X)
# 拿出第一列出来，拿出来是个一维度数组 n 长度 ，再将其列升维，变成 n * 1 的矩阵
# diabetes_X = diabetes_X[:, np.newaxis, 1]
print (diabetes_X)
print(diabetes_X.shape)

# Split the data into training/testing sets
# 将数据分割成 训练集和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]



# Split the targets into training/testing sets

# 将结果分割成 训练结果集和测试结果集
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
# 创建线性回归对象，使用普通最小二乘法 ，最小二乘法在出现multicollinearity 多重共线性（向量之间有高度的关联性）的情况下效果会很差，
regr = linear_model.LinearRegression()
# Train the model using the training sets
# 用训练集来训练模型
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing set
# 使用测试集来做预测
diabetes_y_pred = regr.predict(diabetes_X_test)
# The coefficients
# 获取系数
print('Coefficients: \n', regr.coef_)
# The mean squared error
# 获取均方误差：
# 均方误差 = 偏差+ 方差
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
# 确定系数:1为完美预测
# 使用 r2_score R2 决定系数（你和优度）来评价模型 r2 越好->1，不好->0
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))
# Plot outputs
# 画出曲线
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()


