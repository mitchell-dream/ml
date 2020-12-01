from mxnet import autograd, nd
from mxnet.gluon import nn
import d2lzh as d2l


# 计算二维互相关运算
X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
print(d2l.corr2d(X,K))
# corr2d(X, K)



# 物体边缘检测
# 下面我们来看一个卷积层的简单应用：检测图像中物体的边缘，即找到像素变化的位置。首先我们构造一张 6×8
# 的图像（即高和宽分别为6像素和8像素的图像）。它中间4列为黑（0），其余为白（1）。
X = nd.ones((6, 8))
X[:, 2:6] = 0
print(X)


# 然后我们构造一个高和宽分别为1和2的卷积核K。当它与输入做互相关运算时，如果横向相邻元素相同，输出为0；否则输出为非0。
K = nd.array([[1, -1]])
Y = d2l.corr2d(X, K)
print(Y)

# 构造一个输出通道数为1（将在“多输入通道和多输出通道”一节介绍通道），核数组形状是(1, 2)的二
# 维卷积层
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

# 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)，这里批量大小（批量中的样本数）和通
# 道数均为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # 简单起见，这里忽略了偏差
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

print(conv2d.weight.data().reshape((1, 2)))
