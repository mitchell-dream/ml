import d2lzh as d2l
from mxnet import  autograd, gluon, nd
from mxnet.gluon import  data as gdata, loss as gloss, nn

# 生成数据集
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
featrues = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(featrues, nd.power(featrues, 2), nd.power(featrues,3))
labels = (true_w[0]*poly_features[:,0] + true_w[1]*poly_features[:,1] + true_w[2]*poly_features[:,2] + true_b)
labels += nd.random.normal(scale=0.1, shape=labels.shape)


def semilogy(x_vals, y_vals,x_label, y_label,x2_vals = None, y2_vals=None, legend=None, figsize=(3.5,2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
        d2l.plt.show()

num_epochs, loss = 100, gloss.L2Loss()

def fit_and_plot(train_featrues, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_featrues, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_featrues), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1,num_epochs+1), train_ls, 'epochs', 'loss', range(1,num_epochs+1), test_ls,['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(), '\nbias:', net[0].bias.data().asnumpy())
# 三阶多项式函数拟合（正常）
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:,:], labels[:n_train], labels[n_train:])
# final epoch: train loss 0.0069564153 test loss 0.006310696
# weight: [[ 1.1771805 -3.3922877  5.600545 ]]
# bias: [4.9874463]


# 欠拟合
# 线性模型在非线性模型生成的数据集上很容易欠拟合
fit_and_plot(featrues[:n_train,:], featrues[n_train:,:],labels[:n_train],labels[n_train:])
# final epoch: train loss 159.3326 test loss 102.91548
# weight: [[22.647701]]
# bias: [-0.6612835]


# 过拟合
# 训练样本少，显得模型过于复杂，训练容易被数据中的噪声影响，虽然训练误差低，但是测试误差很高
fit_and_plot(poly_features[0:2,:], poly_features[n_train:,:],labels[0:2],labels[n_train:])
