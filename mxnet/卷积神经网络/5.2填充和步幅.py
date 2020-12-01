from mxnet import nd
from mxnet.gluon import nn

# 定义一个函数来计算卷积层。它初始化卷积层权重，并对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)


# 填充转换公式：
# 一般来说，如果在高的两侧一共填充 Pk 行，在宽的两侧一共填充 Pw 列，那么输出形状将会是
# (Nh - Kh + Ph + 1) * (Nw - Kw + Pw + 1)

# 生成 8 * 8 矩阵
X = nd.random.uniform(shape=(8, 8))
print('X=%s'%X)

print("conv2d=%s"%conv2d)
# 输出的高和宽也是 8*8
print(comp_conv2d(conv2d, X).shape)


# 调整步幅宽高为 2，2
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape

# 调整步幅宽高为 3，4
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape