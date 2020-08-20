#数据处理部分之前的代码，加入部分数据处理的库
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
import gzip
import json
import random

# 上一节，我们通过调用飞桨提供的API（paddle.dataset.mnist）加载MNIST数据集。但在工业实践中，我们面临的任务和数据环境千差万别，通常需要自己编写适合当前任务的数据处理程序，一般涉及如下五个环节：
#
# 读入数据
# 划分数据集
# 生成批次数据
# 训练样本集乱序
# 校验数据有效性
