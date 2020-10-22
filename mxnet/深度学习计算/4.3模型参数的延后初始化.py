
# 4.3.1. 延后初始化
'''
系统将真正的参数初始化延后到获得足够信息时才执行的行为叫作延后初始化。
延后初始化的主要好处是让模型构造更加简单。例如，我们无须人工推测每个层的输入个数。
也可以避免延后初始化。
'''




'''
例如，在上一节使用的多层感知机net里，我们创建的隐藏层仅仅指定了输出大小为256。
当调用initialize函数时，由于隐藏层输入个数依然未知，系统也无法得知该层权重参数的形状。
只有在当我们将形状是(2, 20)的输入X传进网络做前向计算net(X)时，系统才推断出该层的权重参数形状为(256, 20)。
因此，这时候我们才能真正开始初始化参数。
'''

from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化逻辑在此省略了

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

#initialize函数执行完并未打印任何信息。调用initialize函数时并没有真正初始化参数。下面我们定义输入并执行一次前向计算。
net.initialize(init=MyInit())

# 根据输入X做前向计算时，系统能够根据输入的形状自动推断出所有层的权重参数的形状。
# 系统在创建这些参数之后，调用MyInit实例对它们进行初始化，然后才进行前向计算。
X = nd.random.uniform(shape=(2, 20))
# 系统将真正的参数初始化延后到获得足够信息时才执行的行为叫作延后初始化（deferred initialization）。
# 它可以让模型的创建更加简单：只需要定义每个层的输出大小，而不用人工推测它们的输入个数。这对于之后将介绍的定义多达数十甚至数百层的网络来说尤其方便。
# 然而，任何事物都有两面性。延后初始化也可能会带来一定的困惑。
# 在第一次前向计算之前，我们无法直接操作模型参数，例如无法使用data函数和set_data函数来获取和修改参数。
# 因此，我们经常会额外做一次前向计算来迫使参数被真正地初始化。
Y = net(X)

# 初始化只会进行一次，再调用不会再走初始化方法
# Y = net(X)


# 4.3.2. 避免延后初始化

# 第一种情况是我们要对已初始化的模型重新初始化时。因为参数形状不会发生变化，所以系统能够立即进行重新初始化。
net.initialize(init=MyInit(), force_reinit=True)

# 第二种情况是我们在创建层的时候指定了它的输入个数，使系统不需要额外的信息来推测参数形状。
# 下例中我们通过in_units来指定每个全连接层的输入个数，使初始化能够在initialize函数被调用时立即发生。
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))
net.initialize(init=MyInit())