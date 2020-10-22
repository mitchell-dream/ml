from mxnet import autograd, nd

# 使用Gluon可以更简洁地实现模型。
# 在Gluon中，data模块提供了有关数据处理的工具，nn模块定义了大量神经网络的层，loss模块定义了各种损失函数。
# MXNet的initializer模块提供了模型参数初始化的各种方法。




# 3.3.1. 生成数据集
num_inputs = 2
num_exaples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_exaples, num_inputs))
labels = true_w[0]*features[:, 0] + true_w[1] *features[:,1]+true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 3.3.2. 读取数据集
from mxnet.gluon import data as gdata

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break

# 3.3.3. 定义模型
from mxnet.gluon import nn

# nn 是 neural networks（神经网络）的缩写
# Sequential 实例可以看作是一个串联各个层的容器。在构造模型时，我们在该容器中依次添加层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入。
net = nn.Sequential()
# 作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。因此，线性回归的输出层又叫全连接层。在Gluon中，全连接层是一个Dense实例。我们定义该层输出个数为1。
net.add(nn.Dense(1))

# 3.3.4. 初始化模型参数
from mxnet import init
# 指定权重参数每个元素将在初始化时随机采样于均值为0、标准差为0.01的正态分布。偏差参数默认会初始化为零。
net.initialize(init.Normal(sigma=0.01))


# 定义损失函数
# loss模块定义了各种损失函数。我们用假名gloss代替导入的loss模块，并直接使用它提供的平方损失作为模型的损失函数。
from mxnet.gluon import loss as gloss

# 平方损失函数又称 L2 范数损失
loss = gloss.L2Loss()

# 定义优化算法
from mxnet import gluon
# 创建一个Trainer实例，并指定学习率为0.03的小批量随机梯度下降（sgd）为优化算法。
# 该优化算法将用来迭代net实例所有通过add函数嵌套的层所包含的全部参数。这些参数可以通过collect_params函数获取。
trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate':0.03})

# 训练模型
num_epochs = 3
# 通过调用Trainer实例的step函数来迭代模型参数。
# 由于变量l是长度为batch_size的一维NDArray，
# 执行l.backward()等价于执行l.sum().backward()。
# 按照小批量随机梯度下降的定义，我们在step函数中指明批量大小，从而对批量中样本梯度求平均。
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        # l.backward() <=> nd.sum(loss).bacward() 相当于把一个 batch 的loss 都加起来求梯度
        l.backward()
        # trainer.step(batch_size) 相当于把计算出来的梯度 除以 batch_size，弱化在更新参数时的影响
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))


# 分别比较学到的模型参数和真实的模型参数。我们从net获得需要的层，并访问其权重（weight）和偏差（bias）。学到的参数和真实的参数很接近。
dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())

