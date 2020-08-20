import numpy as np
import json
import matplotlib.pyplot as plt

# 读入训练数据
# datafile = 'home/aistudio/data/data16317/housing.data'
# data = np.fromfile(datafile, sep=' ')

# # 数据形状变换，读入的原始数据是一维的，所偶遇数据都连载一起，因此需要我们进行形状变换，形成2维矩阵。
# # 每行维一个数据样本（14个值），每个数据样本包含13 个 X（特征） 1个Y(均价)。

# # 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... 
# # 这里对原始数据做reshape，变成N x 14的形式
# feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
#                  'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
# feature_num = len(feature_names)
# data = data.reshape([data.shape[0] // feature_num, feature_num])


# # 数据集划分,将80%的数据用作训练集，20%用作测试集
# ratio = 0.8
# offset = int(data.shape[0] * ratio)
# training_data = data[:offset]
# training_data.shape

# print(training_data)


# # 数据归一化处理，对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间。
# # 这样做有两个好处：
# # 一是模型训练更高效；
# # 二是特征前的权重大小可以代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。
# # 计算 train 数据集的最大值，最小值，平均值
# maximums, minimums, avgs = \
#                      training_data.max(axis=0), \
#                      training_data.min(axis=0), \
#      training_data.sum(axis=0) / training_data.shape[0]
# # 对数据进行归一化处理
# for i in range(feature_num):
#     #print(maximums[i], minimums[i], avgs[i])
#     data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])



# 数据预处理函数封装
def load_data():
    # 从文件导入数据
    datafile = 'home/aistudio/data/data16317/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# # 1、 获取数据
# training_data, test_data = load_data()
# x = training_data[:, :-1]
# y = training_data[:, -1:]



# 2、模型设计
# 如果将输入特征和输出预测值均以向量表示，输入特征$x$有13个分量，$y$有1个分量，那么参数权重的形状（shape）是$13\times1$。假设我们以如下任意数字赋值参数做初始化：

w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
w = np.array(w).reshape([13, 1])



import numpy as np

class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子，w 给个随机的向量
        #np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
    # 前向计算：计算 w * x + b 的结果
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    # 损失函数，用来衡量模型的好坏
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        # 均方误差
        cost = error * error
        # 需要考虑每个样本的损失，这里求和 并除以样本总数 N
        cost = np.sum(cost) / num_samples
        return cost
    
    # 梯度下降法
    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z-y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z-y)
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
            
                
    def train(self, training_data, num_epoches, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        # 在两层循环的内部是经典的四步训练流程：前向计算->计算损失->计算梯度->更新参数，这与大家之前所学是一致的，代码如下：
        for epoch_id in range(num_epoches):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                #print(self.w.shape)
                #print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                 format(epoch_id, iter_id, loss))
        
        return losses

# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epoches=100, batch_size=200, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

