from Postgresql import *
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')
myfont = fm.FontProperties()

a = Postgresql()
sql = "SELECT * FROM volte.vn_potential_complaint_results WHERE date ='2020-08-08 00:00:00' "
data = a.GetData(sql)
print(data.columns)
print(data.iloc[:, 0].size)
data = pd.DataFrame(data)
data.drop('date', axis=1, inplace=True)
a.finish()
print(data.head())
# 查看是否有空值
print(data.isnull().sum())
# 查看属于投诉统计数据
print('投诉用户')
print(data.comp_count[data.iscomp == 1].describe())
# 查看属于非投诉的统计数据
print('非投诉用户')
print(data.comp_count[data.iscomp == 0].describe())
# 可视化数据
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
bins = 50
ax1.hist(data.comp_count[data.iscomp == 1], bins=bins)
ax1.set_title('投诉用户', fontproperties=myfont)
ax2.hist(data.comp_count[data.iscomp == 0], bins=bins)
ax2.set_title('非投诉用户', fontproperties=myfont)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.xlabel('三个月内投诉次数', fontproperties=myfont)
plt.ylabel('用户数', fontproperties=myfont)
plt.show()

r_data = data.drop(['id', 'iscomp'], axis=1)
r_features = r_data.columns
print(r_data.columns)
##可视化其他特征分布信息
plt.figure(figsize=(12, 14 * 4))
gs = gridspec.GridSpec(14, 1)
for i, cn in enumerate(r_data[r_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(data[cn][data.iscomp == 1], bins=50)
    sns.distplot(data[cn][data.iscomp == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('特征直方图: ' + str(cn), fontproperties=myfont)

plt.show()

# 创建非投诉新特征
data.loc[data.iscomp == 0, 'notcomp'] = 1
data.loc[data.iscomp == 1, 'notcomp'] = 0
data['notcomp'] = data.notcomp.astype(int)
print('样本数据统计')
print(data.notcomp.value_counts())
print(data.iscomp.value_counts())
# 显示的最大行数和列数，如果超额就显示省略号，这个指的是多少个dataFrame的列。
# 如果比较多又不允许换行，就会显得很乱
pd.set_option('display.max_columns', None)  # 显示所有列
print(data.head())
# 创建一个只有投诉、非投诉的Datafram
comp = data[data.iscomp == 1]
notcomp = data[data.iscomp == 0]
# 将train_X设置为投诉样本的80%
train_X = comp.sample(frac=0.8)
count_comp = len(train_X)
print(count_comp)
# 将80%的非投诉样本添加到train_X
train_X = pd.concat([train_X, notcomp.sample(frac=0.8)], axis=0)
print(len(train_X))
# 使test_X包含不在train_X中的数据
test_X = data.loc[~data.index.isin(train_X.index)]
# 使用shuffle函数打乱数据
train_X = shuffle(train_X)
test_X = shuffle(test_X)
print(len(train_X[train_X.iscomp == 0]))
# 把标签添加到train_Y,test_Y
train_Y = train_X.iscomp
train_Y = pd.concat([train_Y, train_X.notcomp], axis=1)

test_Y = test_X.iscomp
test_Y = pd.concat([test_Y, test_X.notcomp], axis=1)

# 删除train_X,test_X中的标签
train_X = train_X.drop(['id', 'iscomp', 'notcomp'], axis=1)
test_X = test_X.drop(['id', 'iscomp', 'notcomp'], axis=1)
# 核查训练集及测试数据所有特征
print(len(train_X))
print(len(train_Y))
print(len(test_X))
print(len(test_Y))
# 提取训练集中所有特征名称
features = train_X.columns.values
print(features)

# 规范化各特征的值
for feature in features:
    mean, std = data[feature].mean(), data[feature].std()
    train_X.loc[:, feature] = (train_X[feature] - mean) / std
    test_X.loc[:, feature] = (test_X[feature] - mean) / std

# 构建1个输入层，2个隐含层，1个输出层的神经网络。隐含层的参数初始化满足正太分布
# 设置参数
learning_rate = 0.005
training_dropout = 0.6
display_step = 1
training_epochs = 5  # 决定迭代次数
batch_size = 100  # 设置批次大小
accuracy_history = []
cost_history = []
valid_accuracy_history = []
valid_cost_history = []
# 获取输入节点数
input_nodes = train_X.shape[1]  # 14
# 设置标签类别数
num_labels = 2
# 把测试数据分为验证集和测试集
split = int(len(test_Y) / 2)

train_size = train_X.shape[0]
n_samples = train_Y.shape[0]
print(train_size, n_samples)
print('123')
print(train_X.head())
input_X = train_X.as_matrix()
input_Y = train_Y.as_matrix()
input_X_valid = test_X.as_matrix()[:split]
input_Y_valid = test_Y.as_matrix()[:split]
input_X_test = test_X.as_matrix()[split:]
input_Y_test = test_Y.as_matrix()[split:]


# 设置每个隐含层的节点数
def calculate_hidden_nodes(nodes):
    return (((2 * nodes) / 3) + num_labels)


hidden_nodes1 = round(calculate_hidden_nodes(input_nodes))
# hidden_nodes2 = round(calculate_hidden_nodes(hidden_nodes1))

print(input_nodes, hidden_nodes1)

# 设置保存进行dropout操作时保留节点的比例变量
pkeep = tf.placeholder(tf.float32)

# 定义输入层# 训练数据的占位符
x = tf.placeholder(tf.float32, [None, input_nodes],name ='X' )

# 定义第一个隐含层layer1，初始化为截断的正太分布
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev=0.15),name ='w1')
b1 = tf.Variable(tf.zeros([hidden_nodes1]),name ='b1')
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y1 = tf.nn.dropout(y1, pkeep,name='y1')

# 定义第二个隐含层layer2，初始化为截断的正太分布
W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, 2], stddev=0.15),name ='w2')
b2 = tf.Variable(tf.zeros([2]),name='b2')
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2,name ='y2')

# 定义输出#训练数据的占位符
y = y2
y_ = tf.placeholder(tf.float32, [None, num_labels],name ='Y_')
argmax_y_ = tf.argmax(y_, 1, name="argmax_y_")
# 使用交叉熵最小化误差
cost = -tf.reduce_sum(y_ * tf.math.log(y))
# 使用Adam作为优化器
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)
# 测试模型# 一样返回True,否则返回False
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算精度，将correct_prediction，转换成指定tf.float32类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化变量
init = tf.compat.v1.global_variables_initializer()

# 保存模型，这里做多保存4个模型
saver = tf.compat.v1.train.Saver(max_to_keep=4)
path = 'D:\PycharmProjects\TSFL\poten_model'
# print(input_X)
# 训练模型迭代次数有参数training_epochs确定，批次大小由参数batch_size确定
# 定义会话
with tf.compat.v1.Session() as sess:
    # 所有变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(training_epochs):
        for batch in range(int(n_samples / batch_size)):
            batch_x = input_X[batch * batch_size: (1 + batch) * batch_size]
            batch_y = input_Y[batch * batch_size: (1 + batch) * batch_size]
            # # print(len(batch_x),len(batch_y))
            # print('batch_x')
            # print(batch_x)
            # print('batch_y')
            # print(batch_y)

            sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y, pkeep: training_dropout})

        # 循环10次打印日志信息
        if (epoch) % display_step == 0:
            train_accuracy, newCost = sess.run([accuracy, cost],
                                               feed_dict={x: input_X, y_: input_Y, pkeep: training_dropout})

            valid_accuracy, valid_newCost = sess.run([accuracy, cost],
                                                     feed_dict={x: input_X_valid, y_: input_Y_valid, pkeep: 1})

            print("Epoch:", epoch, "Acc =", "{:.5f}".format(train_accuracy), "Cost =", "{:.5f}".format(newCost),
                  "Valid_Acc =", "{:.5f}".format(valid_accuracy), "Valid_Cost = ", "{:.5f}".format(valid_newCost))

            # 记录模型结果
            accuracy_history.append(train_accuracy)
            cost_history.append(newCost)
            valid_accuracy_history.append(valid_accuracy)
            valid_cost_history.append(valid_newCost)

            # 如若15次日志信息没有改善，停止迭代
            if valid_accuracy < max(valid_accuracy_history) and epoch > 100:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0

            if valid_accuracy > 0.95:
                # 保存模型
                saver.save(sess, path + "/model1", global_step=epoch)
    print("Optimization Finished!")
    # 可视化精度及损失值
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

    ax1.plot(accuracy_history, color='b')
    ax1.plot(valid_accuracy_history, color='r')
    ax1.set_title('精度', fontproperties=myfont)

    ax2.plot(cost_history, color='b')
    ax2.plot(valid_cost_history, color='r')
    ax2.set_title('损失值', fontproperties=myfont)

    plt.xlabel('迭代次数 (x10)', fontproperties=myfont)
    plt.show()

sess.close()

#################################################################
#测试模型
#####加载模型及权重
# saver = tf.train.import_meta_graph(path+'model1-1.meta')
# saver.restore(sess, tf.train.latest_checkpoint(path))
# sess = tf.Session()
