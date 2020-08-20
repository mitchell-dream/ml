from mxnet import autograd, nd

# 生成数据及
num_inputs = 2
num_exaples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_exaples, num_inputs))
labels = true_w[0]*features[:, 0] + true_w[1] *features[:,1]+true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 读取数据集
from mxnet.gluon import data as gdata

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break



# 定义模型
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))

# 初始化模型参数
from mxnet import init
net.initialize(init.Normal(sigma=0.01))


# 定义损失函数
from mxnet.gluon import loss as gloss

# 平方损失函数
loss = gloss.L2Loss()


# 定义优化算法
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate':0.03})

# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' %(epoch, l.mean().asumpy()))





