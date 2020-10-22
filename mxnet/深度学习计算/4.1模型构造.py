from mxnet import nd
from mxnet.gluon import nn

# 4.1.1. 继承Block类来构造模型
# 自定义 MLP 类
class MLP(nn.Block):
    # 声明带有模型参数的层，这里声明了两个全连接层
    # 重写 init 方法
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 隐藏层
        self.output = nn.Dense(10)  # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    # 重写前向计算方法
    def forward(self, x):
        return self.output(self.hidden(x))
    # 不定义反向传播函数，系统将通过自动求梯度而自动生成反向传播所需的 backward 函数


# 实例化 MLP 模型变量
X = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
# net() 方法自动调用 继承 Block 类的 MLP 的 __call__函数，这个函数将调用MLP类定义的forward函数来完成前向计算
net(X)


# 4.1.2. Sequential类继承自Block类

# Block 类是一个通用的部件。Sequential 类继承自 Block 类。
# 当模型的前向计算为简单串联各个层的计算时，可以通过更加简单的方式定义模型。
# 这正是Sequential类的目的：它提供add函数来逐一添加串联的Block子类实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # block是一个Block子类实例，假设它有一个独一无二的名字。我们将它保存在Block类的
        # 成员变量_children里，其类型是OrderedDict。当MySequential实例调用
        # initialize函数时，系统会自动对_children里所有成员初始化
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict保证会按照成员添加时的顺序遍历成员
        for block in self._children.values():
            x = block(x)
        return x


# 用 MySequential 类来实现前面描述的MLP类，并使用随机初始化的模型做一次前向计算。
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)

# 4.1.3. 构造复杂的模型
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用get_constant创建的随机权重参数不会在训练中被迭代（即常数参数）
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常数参数，以及NDArrrand_weightay的relu函数和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层。等价于两个全连接层共享参数
        x = self.dense(x)
        # 控制流，这里我们需要调用asscalar函数来返回标量进行比较
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()
# 测试模型
net = FancyMLP()
net.initialize()
net(X)


# 因为FancyMLP和Sequential类都是Block类的子类，所以我们可以嵌套调用它们。
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())
net.initialize()
net(X)