from Postgresql import *
import tensorflow as tf
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
##获取预测数据
a = Postgresql()
sql = "SELECT * FROM volte.vn_potential_complaint_results WHERE date ='2020-08-08 00:00:00' LIMIT 100"
data = a.GetData(sql)
print(data.columns)
print(data.iloc[:, 0].size)
data = pd.DataFrame(data)
a.finish()
print(data.head())

data.drop(['id', 'iscomp', 'date'], axis=1, inplace=True)
data = data.astype(np.float32)
# 查看是否有空值
print(data.isnull().sum())
long = len(data.comp_count)
# 提取预测数据中所有特征名称
features = data.columns.values
# 规范化各特征的值
for feature in features:
    mean, std = data[feature].mean(), data[feature].std()
    data.loc[:, feature] = (data[feature] - mean) / std

pd.set_option('display.max_columns', None)  # 显示所有列
# print(data.head())

path = 'D:\PycharmProjects\TSFL\poten_model/'
# XX = data.as_matrix()
XX = data.values
XX[np.isnan(data)] = 0.0
ten = torch.from_numpy(XX)


# 测试模型
#####加载模型及权重
print(ten)

saver = tf.compat.v1.train.import_meta_graph(path + 'model1-0.meta')
batch_size = 100  # 设置批次大小
n_samples = data.shape[0]
with tf.compat.v1.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(path))
    feed_dict = {"X:0": ten}
    graph = tf.compat.v1.get_default_graph()

    # print(tf.get_default_graph().as_graph_def())
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    print(tensor_name_list)



    y = graph.get_tensor_by_name("y2:0")
    # # # 根据需要配置输出
    # 通过张量名称获得张量
    print(sess.run(y,feed_dict))

