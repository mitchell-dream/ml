# 4.5.1. 读写NDArray
from mxnet import nd
from mxnet.gluon import nn
'''
通过save函数和load函数可以很方便地读写NDArray。
通过load_parameters函数和save_parameters函数可以很方便地读写Gluon模型的参数
'''
# 创建了NDArray变量x
x = nd.ones(3)
# 存在文件名同为x的文件里。
nd.save('x', x)
x2 = nd.load('x')
print(x2)

# 存储一列NDArray并读回内存。
y = nd.zeros(4)
nd.save('xy', [x, y])
x2, y2 = nd.load('xy')
print(x2, y2)

# 存储并读取一个从字符串映射到NDArray的字典
mydict = {'x': x, 'y': y}
nd.save('mydict', mydict)
mydict2 = nd.load('mydict')
print(mydict2)


# 4.5.2. 读写Gluon模型的参数

# Gluon的Block类提供了save_parameters函数和load_parameters函数来读写模型参数。
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = nd.random.uniform(shape=(2, 20))
Y = net(X)

filename = 'mlp.params'
net.save_parameters(filename)

net2 = MLP()
net2.load_parameters(filename)

Y2 = net2(X)
print(Y2 == Y)