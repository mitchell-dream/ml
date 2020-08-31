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


def init_params():
    w = nd.random.normal(scale=1,shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w,b]

def l2_penalt(w):
    return (w**2).sum() / 2

batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalt(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().asscalar())

# 未使用权重衰减 产生 过拟合现象
fit_and_plot(lambd=0)
# L2 norm of w: 13.155678

# 使用权重衰减(正则化)，减少了过拟合的现象
fit_and_plot(lambd=3)
# L2 norm of w: 0.042778388
