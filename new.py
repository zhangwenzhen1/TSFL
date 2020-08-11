#导入需要的库或模块
import theano
from theano import tensor as T
import numpy  as np
# #初始化张量
# x=T.scalar(name='input',dtype='float32')
# w=T.scalar(name='weight',dtype='float32')
# b=T.scalar(name='bias',dtype='float32')
# z=w*x+b
#
# #编译程序
# net_input=theano.function(inputs=[w,x,b],outputs=z)
# #执行程序
# print('net_input: %2f'% net_input(2.0,3.0,0.5))

rng = np.random
# 我们为了测试，自己生成10个样本，每个样本是3维的向量，然后用于训练
N = 10
feats = 3
D = (rng.randn(N, feats).astype(np.float32), rng.randint(size=N, low=0, high=2).astype(np.float32))
# 声明自变量x、以及每个样本对应的标签y(训练标签)
x = T.matrix("x")
y = T.vector("y")
#随机初始化参数w、b=0，为共享变量
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
#构造代价函数
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # s激活函数
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # 交叉商代价函数
cost = xent.mean() + 0.01 * (w ** 2).sum()# 代价函数的平均值+L2正则项以防过拟合，其中权重衰减系数为0.01
gw, gb = T.grad(cost, [w, b])             #对总代价函数求参数的偏导数

prediction = p_1 > 0.5                    # 大于0.5预测值为1，否则为0.

train = theano.function(inputs=[x,y],outputs=[prediction, xent],updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))#训练所需函数
predict = theano.function(inputs=[x], outputs=prediction)#测试阶段函数

#训练
training_steps = 1000
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print (err.mean())#查看代价函数下降变化过程

