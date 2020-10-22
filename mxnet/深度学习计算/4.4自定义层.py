from mxnet import gluon, nd
from mxnet.gluon import nn
'''
通过Block类自定义神经网络中的层，从而可以被重复调用。
'''

# 4.4.1. 不含模型参数的自定义层
# CenteredLayer 类通过继承 Block 类自定义了一个将输入减掉均值后输出的层，并将层的计算定义在了forward函数里。这个层里不含模型参数。
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
# 实例化这个层，然后做前向计算。
layer = CenteredLayer()
print(layer(nd.array([1, 2, 3, 4, 5])))

# 构建更复杂的模型
net = nn.Sequential()
net.add(nn.Dense(128),
        CenteredLayer())

# 下面打印自定义层各个输出的均值。因为均值是浮点数，所以它的值是一个很接近0的数。
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
print(y.mean().asscalar())

# 4.4.2. 含模型参数的自定义层
'''
分别介绍了Parameter类和ParameterDict类。
在自定义含模型参数的层时，我们可以利用Block类自带的ParameterDict类型的成员变量params。
它是一个由字符串类型的参数名字映射到Parameter类型的模型参数的字典。我们可以通过get函数从ParameterDict创建Parameter实例。
'''
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
print(params)

# 尝试实现一个含权重参数和偏差参数的全连接层。
# 它使用ReLU函数作为激活函数。其中in_units和units分别代表输入个数和输出个数。
class MyDense(nn.Block):
    # units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)

# 实例化 MyDense
dense = MyDense(units=3, in_units=5)
print(dense.params)


# 可以直接使用自定义层做前向计算。
dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))

# 自定义层构造模型。
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))