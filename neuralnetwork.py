import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

myfont = fm.FontProperties()
train_filename = "data.csv"
idKey = "id"
diagnosisKey = "diagnosis"
radiusMeanKey = "radius_mean"
textureMeanKey = "texture_mean"
perimeterMeanKey = "perimeter_mean"
areaMeanKey = "area_mean"
smoothnessMeanKey = "smoothness_mean"
compactnessMeanKey = "compactness_mean"
concavityMeanKey = "concavity_mean"
concavePointsMeanKey = "concave points_mean"
symmetryMeanKey = "symmetry_mean"
fractalDimensionMean = "fractal_dimension_mean"
radiusSeKey = "radius_se"
textureSeKey = "texture_se"
perimeterSeKey = "perimeter_se"
areaSeKey = "area_se"
smoothnessSeKey = "smoothness_se"
compactnessSeKey = "compactness_se"
concavitySeKey = "concavity_se"
concavePointsSeKey = "concave points_se"
symmetrySeKey = "symmetry_se"
fractalDimensionSeKey = "fractal_dimension_se"
radiusWorstKey = "radius_worst"
textureWorstKey = "texture_worst"
perimeterWorstKey = "perimeter_worst"
areaWorstKey = "area_worst"
smoothnessWorstKey = "smoothness_worst"
compactnessWorstKey = "compactness_worst"
concavityWorstKey = "concavity_worst"
concavePointsWorstKey = "concave points_worst"
symmetryWorstKey = "symmetry_worst"
fractalDimensionWorstKey = "fractal_dimension_worst"
train_columns = [idKey, diagnosisKey, radiusMeanKey, textureMeanKey, perimeterMeanKey, areaMeanKey, smoothnessMeanKey,
                 compactnessMeanKey, concavityMeanKey, concavePointsMeanKey, symmetryMeanKey, fractalDimensionMean,
                 radiusSeKey, textureSeKey, perimeterSeKey, areaSeKey, smoothnessSeKey, compactnessSeKey,
                 concavitySeKey, concavePointsSeKey, symmetrySeKey, fractalDimensionSeKey, radiusWorstKey,
                 textureWorstKey, perimeterWorstKey, areaWorstKey, smoothnessWorstKey, compactnessWorstKey,
                 concavityWorstKey, concavePointsWorstKey, symmetryWorstKey, fractalDimensionWorstKey]


def get_train_data():
    df = pd.read_csv(train_filename, names=train_columns, delimiter=',', skiprows=1)
    return df


train_data = get_train_data()
print(train_data.head())
# 查看是否有空值
print(train_data.isnull().sum())
# 查看属于恶性的统计数据
print('恶性')
print(train_data.area_mean[train_data.diagnosis == "M"].describe())
# 查看属于良性的统计数据
print('良性')
print(train_data.area_mean[train_data.diagnosis == "B"].describe())
# 可视化数据
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
bins = 50
ax1.hist(train_data.area_mean[train_data.diagnosis == "M"], bins=bins)
ax1.set_title('恶性', fontproperties=myfont)
ax2.hist(train_data.area_mean[train_data.diagnosis == "B"], bins=bins)
ax2.set_title('良性', fontproperties=myfont)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.xlabel('区域平均值', fontproperties=myfont)
plt.ylabel('诊断次数', fontproperties=myfont)
plt.show()

r_data = train_data.drop([idKey, areaMeanKey, areaWorstKey, diagnosisKey], axis=1)
r_features = r_data.columns
print(r_data.columns)

##可视化其他特征分布信息
plt.figure(figsize=(12, 28 * 4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(r_data[r_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(train_data[cn][train_data.diagnosis == "M"], bins=50)
    sns.distplot(train_data[cn][train_data.diagnosis == "B"], bins=50)
    ax.set_xlabel('')
    ax.set_title('特征直方图: ' + str(cn), fontproperties=myfont)

plt.show()

# 更新诊断值。1代表恶性，0代表良性
train_data.loc[train_data.diagnosis == "M", 'diagnosis'] = 1
train_data.loc[train_data.diagnosis == "B", 'diagnosis'] = 0
# 创建良性(非恶性)新特征
train_data.loc[train_data.diagnosis == 0, 'benign'] = 1
train_data.loc[train_data.diagnosis == 1, 'benign'] = 0

train_data['benign'] = train_data.benign.astype(int)
train_data = train_data.rename(columns={'diagnosis': 'malignant'})

# print(train_data.benign.value_counts())
# print()
# print(train_data.malignant.value_counts())
# 显示的最大行数和列数，如果超额就显示省略号，这个指的是多少个dataFrame的列。
# 如果比较多又不允许换行，就会显得很乱
# pd.set_option('display.max_columns', None)  # 显示所有列
# print(train_data.head())
# 创建一个只有malignant、benign的Datafram
Malignant = train_data[train_data.malignant == 1]
Benign = train_data[train_data.benign == 1]
# 将train_X设置为恶性诊断的80%
train_X = Malignant.sample(frac=0.8)
count_Malignants = len(train_X)
print(count_Malignants)
# 将80%的良性诊断结果添加到train_X
train_X = pd.concat([train_X, Benign.sample(frac=0.8)], axis=0)
print(len(train_X))
# 使test_X包含不在train_X中的数据
test_X = train_data.loc[~train_data.index.isin(train_X.index)]
# 使用shuffle函数打乱数据
train_X = shuffle(train_X)
test_X = shuffle(test_X)
# 把标签添加到train_Y,test_Y
train_Y = train_X.malignant
train_Y = pd.concat([train_Y, train_X.benign], axis=1)

test_Y = test_X.malignant
test_Y = pd.concat([test_Y, test_X.benign], axis=1)
# 删除train_X,test_X中的标签
train_X = train_X.drop(['malignant', 'benign'], axis=1)
test_X = test_X.drop(['malignant', 'benign'], axis=1)
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
    mean, std = train_data[feature].mean(), train_data[feature].std()
    train_X.loc[:, feature] = (train_X[feature] - mean) / std
    test_X.loc[:, feature] = (test_X[feature] - mean) / std

# 构建1个输入层，4个隐含层，1个输出层的神经网络。隐含层的参数初始化满足正太分布
# 设置参数
learning_rate = 0.005
training_dropout = 0.9
display_step = 1
training_epochs = 5
batch_size = 100
accuracy_history = []
cost_history = []
valid_accuracy_history = []
valid_cost_history = []
# 获取输入节点数
input_nodes = train_X.shape[1]  # 31
# 设置标签类别数
num_labels = 2
# 把测试数据分为验证集和测试集
split = int(len(test_Y) / 2)

train_size = train_X.shape[0]
n_samples = train_Y.shape[0]

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
hidden_nodes2 = round(calculate_hidden_nodes(hidden_nodes1))
hidden_nodes3 = round(calculate_hidden_nodes(hidden_nodes2))
print(input_nodes, hidden_nodes1, hidden_nodes2, hidden_nodes3)

# 设置保存进行dropout操作时保留节点的比例变量
pkeep = tf.placeholder(tf.float32)
# 定义输入层# 训练数据的占位符
x = tf.placeholder(tf.float32, [None, input_nodes])
# 定义第一个隐含层layer1，初始化为截断的正太分布
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev=0.15))
b1 = tf.Variable(tf.zeros([hidden_nodes1]))
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# 定义第二个隐含层layer2，初始化为截断的正太分布
W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev=0.15))
b2 = tf.Variable(tf.zeros([hidden_nodes2]))
y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
# 定义第三个隐含层layer3，初始化为截断的正太分布
W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev=0.15))
b3 = tf.Variable(tf.zeros([hidden_nodes3]))
y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
y3 = tf.nn.dropout(y3, pkeep)
# 定义第四个隐含层layer4，初始化为截断的正太分布
W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev=0.15))
b4 = tf.Variable(tf.zeros([2]))
y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)
# 定义输出#训练数据的占位符
y = y4
y_ = tf.placeholder(tf.float32, [None, num_labels])
#使用交叉熵最小化误差
cost = -tf.reduce_sum(y_ * tf.math.log(y))
#使用Adam作为优化器
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)
#测试模型# 一样返回True,否则返回False
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#计算精度，将correct_prediction，转换成指定tf.float32类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#初始化变量
init = tf.compat.v1.global_variables_initializer()

# 保存模型，这里做多保存4个模型
saver = tf.compat.v1.train.Saver(max_to_keep=4)
path = 'D:\PycharmProjects\TSFL\model'
#训练模型迭代次数有参数training_epochs确定，批次大小由参数batch_size确定
# 定义会话
with tf.compat.v1.Session() as sess:
    # 所有变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(training_epochs):
        for batch in range(int(n_samples / batch_size)):
            batch_x = input_X[batch * batch_size: (1 + batch) * batch_size]
            batch_y = input_Y[batch * batch_size: (1 + batch) * batch_size]

            sess.run([optimizer], feed_dict={x: batch_x,
                                             y_: batch_y,
                                             pkeep: training_dropout})

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

            # If the model does not improve after 15 logs, stop the training.
            if valid_accuracy < max(valid_accuracy_history) and epoch > 100:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0

    print("Optimization Finished!")
    # 可视化精度及损失值
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

    ax1.plot(accuracy_history, color='b')
    ax1.plot(valid_accuracy_history, color='g')
    ax1.set_title('精度', fontproperties=myfont)

    ax2.plot(cost_history, color='b')
    ax2.plot(valid_cost_history, color='g')
    ax2.set_title('损失值', fontproperties=myfont)

    plt.xlabel('迭代次数 (x10)', fontproperties=myfont)
    plt.show()
    # 保存模型
    saver.save(sess, path + "/model1", global_step=epoch)
sess.close()