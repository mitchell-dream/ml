import d2lzh as d2l
from mxnet import  autograd, gluon,init, nd
from mxnet.gluon import  data as gdata, loss as gloss, nn


n_train,n_test, num_inputs = 20, 100, 200
true_w, true_b= nd.ones((num_inputs,1)) *0.01, 0.05

featrues = nd.random.normal(shape=(n_train+n_test, num_inputs))
labels = nd.dot(featrues, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = featrues[:n_train, :], featrues[n_train:, :]
train_labels,test_labels = labels[:n_train],labels[n_train:]



batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)



def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    #对权重参数衰减，权重名称一般是以 weight 结尾
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {'learning_rate': lr, 'wd':wd})
    # 不对偏差参数衰减，偏差名称一般是以 bias 结尾
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate': lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net[0].weight.data().norm().asscalar())
# 未使用权重衰减（正则化）
fit_and_plot_gluon(0)
# 使用权重衰减（正则化）
fit_and_plot_gluon(3)