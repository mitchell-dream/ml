# !nvidia-smi  # 对Linux/macOS用户有效


'''
如使用内存的CPU或者使用显存的GPU。
默认情况下，MXNet会将数据创建在内存，然后利用CPU来计算。
在MXNet中，mx.cpu()（或者在括号里填任意整数）表示所有的物理CPU和内存。
这意味着，MXNet 的计算会尽量使用所有的CPU核。但mx.gpu()只代表一块GPU和相应的显存。如果有多块GPU，我们用mx.gpu(i)来表示第 i
 块GPU及相应的显存（ i从0开始）且mx.gpu(0)和mx.gpu()等价
'''
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

print(mx.cpu(), mx.gpu(), mx.gpu(1))

# 4.6.2. NDArray的GPU计算
# 默认情况下，NDArray存在内存上
x = nd.array([1, 2, 3])
print(x)

# NDArray的context属性来查看该NDArray所在的设备。
print(x.context)

# 4.6.2.1. GPU上的存储
'''
我们有多种方法将NDArray存储在显存上。
例如，我们可以在创建NDArray的时候通过ctx参数指定存储设备。
下面我们将NDArray变量a创建在gpu(0)上。注意，在打印a时，设备信息变成了@gpu(0)。
创建在显存上的NDArray只消耗同一块显卡的显存。
我们可以通过nvidia-smi命令查看显存的使用情况。
通常，我们需要确保不创建超过显存上限的数据。
'''

a = nd.array([1, 2, 3], ctx=mx.gpu())
print(a)

# 至少有2块GPU，下面代码将会在gpu(1)上创建随机数组。
# B = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1))
# print(B)

# 通过 copyto 函数和as_in_context函数在设备之间传输数据。下面我们将内存上的NDArray变量x复制到gpu(0)上
y = x.copyto(mx.gpu())
print(y)

z = x.as_in_context(mx.gpu())
print(z)

# 如果源变量和目标变量的 context 一致，as_in_context 函数使目标变量和源变量共享源变量的内存或显存。
print(y.as_in_context(mx.gpu()) is y)

# copyto 函数总是为目标变量开新的内存或显存
print(y.copyto(mx.gpu()) is y)

# 4.6.2.2. GPU上的计算
print((z + 2).exp() * y)


# 4.6.3. Gluon的GPU计算¶
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())
net(y)

net[0].weight.data()