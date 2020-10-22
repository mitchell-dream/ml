import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数


train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

# print(train_data.shape)
# print(test_data.shape)


# 让我们来查看前4个样本的前4个特征、后2个特征和标签（SalePrice）：
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])


print(train_data.iloc[:, 1:-1])
print(test_data.iloc[:, 1:])

# 合并训练数据的 第1列到倒数第二列，测试数据的第一列到最后一列数据
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 特征
print(all_features)

# 特征标准化
# 获取所有特征的 index
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print(numeric_features)
# 对所有特征标准化
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

print(all_features)
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 离散数值转成指示特征
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features)

# 获取训练特征数量
n_train = train_data.shape[0]
# 获取训练特征
train_features = nd.array(all_features[:n_train].values)
# 获取测试特征
test_features = nd.array(all_features[n_train:].values)
# 将矩阵重组成 1 列，行数 -1 代表自动计算行数
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))


# 使用基本的线性回归模型来训练模型
loss = gloss.L2Loss()
loss = d2l.squared_loss
def get_net():
    net = nn.Sequential()
    # net.add(nn.Dense(1))
    # net.initialize()
    drop_prob1, drop_prob2 = 0.2, 0.5
    net.add(
        nn.Dense(500, activation="relu"),
        nn.Dense(1)
    )
    net.initialize(init.Normal(sigma=1))
    return net

# 对数均方误差  根据均方误差来评估模型
def log_rmse(net, features, labels):
    # 将小于1的值设成1，使得取对数时数值更稳定
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()

# 训练使用 Adam 算法进行训练
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# k 折交叉验证，训练 k 次，并返回训练和验证的平均误差
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k
# 5-fold validation: avg train rmse 0.072945, avg valid rmse 0.133392  5, 100, 0.15, 200, 64
# train rmse 0.069722

# 5-fold validation: avg train rmse 0.069804, avg valid rmse 0.135053 5, 100, 0.12, 200, 64
# train rmse 0.067479

# 5-fold validation: avg train rmse 0.075872, avg valid rmse 0.137785 5, 100, 0.12, 200, 64
# train rmse 0.075963

# 模型选择：可以进行超参的调节来尽可能减少平均测试误差
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.15, 200, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
      % (k, train_l, valid_l))


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)