from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
# activation 激活函数 为 relu
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # 使用默认初始化方式

X = nd.random.uniform(shape=(2, 20))
Y = net(X)  # 前向计算


# 4.2.1. 访问模型参数
# 对于使用Sequential类构造的神经网络，我们可以通过方括号[]来访问网络的任一层。
# Sequential类与Block类的继承关系。对于Sequential实例中含模型参数的层，
# 我们可以通过Block类的params属性来访问该层包含的所有参数。
# 访问多层感知机net中隐藏层的所有参数。索引0表示隐藏层为Sequential实例最先添加的层。
print(net[0].params, type(net[0].params))

# 访问权重，根据我们的初始化，权重参数是一个由随机数组成的形状为(256, 20)的NDArray
print(net[0].weight.data())

# 还没有进行反向传播计算，所以梯度的值全为0。
print(net[0].weight.grad())


# 访问输出层的偏差
print(net[1].bias.data())

# collect_params函数来获取net变量所有嵌套（例如通过add函数嵌套）的层所包含的所有参数。
# 它返回的同样是一个由参数名称到参数实例的字典。
print(net.collect_params())

# 这个函数可以通过正则表达式来匹配参数名，从而筛选需要的参数。
print(net.collect_params('.*weight'))

# 4.2.2. 初始化模型参数
# 将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零。
# 非首次对模型初始化需要指定 force_reinit为真
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data()[0])

# 使用常数来初始化权重参数
net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data()[0])

# 只想对某个特定参数进行初始化，我们可以调用 Parameter 类的 initialize 函数，它与Block类提供的initialize函数的使用方法一致。
# 下例中我们对隐藏层的权重使用 Xavier 随机初始化方法。
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[0].weight.data()[0])


# 4.2.3. 自定义初始化方法
# 我们需要的初始化方法并没有在init模块中提供。这时，可以实现一个Initializer类的子类
# 只需要实现 _init_weight 这个函数，并将其传入的NDArray修改成初始化的结果
# 在下面的例子里，我们令权重有一半概率初始化为0，有另一半概率初始化为
# [−10,−5] 和 [5,10] 两个区间里均匀分布的随机数
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5
net.initialize(MyInit(), force_reinit=True)
print(net[0].weight.data()[0])

# 通过 Parameter 类的 set_data 函数来直接改写模型参数。例如，在下例中我们将隐藏层参数在现有的基础上加1。
net[0].weight.set_data(net[0].weight.data() + 1)
print(net[0].weight.data()[0])

# 4.2.4. 共享模型参数
'''
在有些情况下，我们希望在多个层之间共享模型参数,
这里再介绍另外一种方法，它在构造层的时候指定使用特定的参数。
如果不同层使用同一份参数，那么它们在前向计算和反向传播时都会共享相同的参数。在下面的例子里，我们让模型的第二隐藏层（shared变量）和第三隐藏层共享模型参数。
'''
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(
        #第一层
        nn.Dense(8, activation='relu'),
        #第二层
        shared,
        #第三层，第三隐藏层时通过params来指定它使用第二隐藏层的参数。
        # 模型参数中包含了梯度，所以在反向传播计算时，第二隐藏层和第三隐藏层的梯度都会被累加在 shared.params.grad() 中
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = nd.random.uniform(shape=(2, 20))
net(X)

net[1].weight.data()[0] == net[2].weight.data()[0]