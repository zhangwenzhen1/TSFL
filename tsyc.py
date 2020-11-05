# encoding=utf-8
from Postgresql import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
myfont = fm.FontProperties()

class potential(object):
    def __init__(self, data):
        self.data = data

    def choosedata(self):
        data = self.data
        data.fillna(0, inplace=True)
        # 投诉用户
        comp = data.loc[data.iscomp == 1]
        print(comp.iloc[:, 0].size)
        # 非投诉用户
        notcomp = data.loc[data.iscomp == 0]
        print(notcomp.iloc[:, 0].size)
        # 筛选当日未投诉但三个月内投诉过的用户
        copm_temp = data.loc[(data['iscomp'] != 1) & (data['copm_count'] > 0)]
        print("当日未投诉但三个月内投诉过的用户样本数:{}".format(len(copm_temp)))
        # 将train_X设置为投诉用户的80%
        train_X = comp.sample(frac=0.8)
        print("投诉样本数:{}".format(len(train_X)))
        # 将1%的非投诉样本添加到train_X
        train_X = pd.concat([train_X, notcomp.sample(frac=0.01)], axis=0)
        print("总样本数:{}".format(len(train_X)))
        # 使test_X包含不在train_X中的数据
        test_X1 = comp.loc[~comp.index.isin(train_X.index)]
        print("测试投诉样本数:{}".format(len(test_X1)))
        test_X2 = notcomp.loc[~notcomp.index.isin(train_X.index)].sample(frac=0.02)
        print("测试非投诉样本数:{}".format(len(test_X2)))
        test_X = pd.concat([test_X1, test_X2], axis=0)
        # 使用shuffle函数打乱数据
        test_X = shuffle(test_X)
        print("总测试样本数:{}".format(len(test_X)))
        # 添加当日未投诉但三个月内投诉过的用户,避免训练集中未筛选到此类样本
        train_X = pd.concat([train_X, copm_temp], axis=0)
        print("总样本数:{}".format(len(train_X)))
        # 使用shuffle函数打乱数据
        train_X = shuffle(train_X)
        ########################################################
        train_X.drop(['id', 'cgi1', 'cgi2', 'date', 'yewu_type'], axis=1, inplace=True)
        test_X.drop(['id', 'cgi1', 'cgi2', 'date', 'yewu_type'], axis=1, inplace=True)
        traindata = train_X
        return traindata, test_X

    def fig_Picture(self, traindata):
        # 可视化数据
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
        bins = 50
        ax1.hist(traindata['copm_count'].loc[traindata.iscomp == 1], bins=bins)
        ax1.set_title('投诉用户', fontproperties=myfont)
        ax2.hist(traindata['copm_count'].loc[traindata.iscomp == 0], bins=bins)
        ax2.set_title('非投诉用户', fontproperties=myfont)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        plt.xlabel('三个月内投诉次数', fontproperties=myfont)
        plt.ylabel('用户数', fontproperties=myfont)
        plt.show()

        r_data = traindata.drop(['iscomp'], axis=1)
        r_features = r_data.columns
        # print(r_data.columns)
        # 可视化其他特征分布信息
        plt.figure(figsize=(12, 28 * 4))
        gs = gridspec.GridSpec(len(r_data.columns), 1)
        for i, cn in enumerate(r_data[r_features]):
            ax = plt.subplot(gs[i])
            sns.distplot(traindata[cn][traindata.iscomp == 1], bins=50)
            sns.distplot(traindata[cn][traindata.iscomp == 0], bins=50)
            ax.set_xlabel('')
            ax.set_title('特征直方图: ' + str(cn), fontproperties=myfont)

        plt.show()

    def dataNormalized(self, traindata):
        # 提取预测数据中所有特征名称
        features = traindata.columns.values
        # 规范化各特征的值
        for feature in features:
            mean, std = traindata[feature].mean(), traindata[feature].std()
            traindata.loc[:, feature] = (traindata[feature] - mean) / std
        # print(traindata.columns)
        return traindata

    def split_Train_TestData(self, X, y, b):
        """
        # 获取最优参数
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
        # 设置模型参数
        params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200),
                  'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
        # 使用GridSearchCV搜索最优参数的解
        xgbc_best = XGBClassifier(feature_names=b)

        xgbc_best.fit(X_train, y_train)
        plot_importance(xgbc_best, importance_type='gain', title='特征重要性排序', )
        plt.show()

        gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
        gs.fit(X_train, y_train)
        # 显示数据集的结果
        print(gs.best_score_)
        print(gs.best_params_)

    def ModelTrain(self, X, y, b):
        # 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123457)
        # 算法参数
        params = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',  # 多分类的问题 指定学习任务和相应的学习目
            'num_class': 2,  # 类别数，多分类与 multisoftmax 并
            'gamma': 0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子
            'max_depth': 6,  # 构建树的深度，越大越容易过拟合
            'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
            'subsample': 0.7,  # 随机采样训练样本，训练实例的子采样比
            'colsample_bytree': 0.7,  # 生成树时进行的列采样
            'min_child_weight': 3,  # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言假设 h
            # 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。这个参数非常影响结果，控制叶子节点中
            # 二阶导的和的最小值，该参数值越小，越容易 overfitting
            # 'slient': 1,
            'eta': 0.1,  # 学习率
            'seed': 1000,  # 随机种子
            'nthread': 4,  # cpu 线程数 默认最大
        }

        plst = list(params.items())

        # 生成数据集格式
        dtrain = xgb.DMatrix(X_train, y_train, feature_names=b)
        num_rounds = 500
        # xgboost模型训练
        model = xgb.train(plst, dtrain, num_rounds)
        # 解决中文和负号显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 使显示图标自适应
        plt.rcParams['figure.autolayout'] = True
        # 显示重要特征
        plot_importance(model, importance_type='gain', title='特征重要性排序', )
        plt.show()

        # 对测试集进行预测
        dtest = xgb.DMatrix(X_test, feature_names=b)
        y_pred = model.predict(dtest)
        # print(y_pred)
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print('accuarcy:%.2f%%' % (accuracy * 100))

        importance = model.get_fscore()
        print(model.get_fscore())
        my_df = pd.DataFrame(importance, index=[0])
        # print(my_df.head())
        # print(my_df.columns)
        return my_df.columns, model

    # 保存模型
    def save_model(self, model, path, model_name):
        model.save_model(path + '/' + model_name)  # 用于存储训练出的模型

    # 加载模型
    def load_model(self, path, model_name):
        tar = xgb.Booster(model_file=path + '/' + model_name)
        return tar

    def model_predict(self, tar, test, ytest):
        dtest = xgb.DMatrix(test)
        preds = tar.predict(dtest)
        # 计算准确率
        test_accuracy = accuracy_score(ytest, preds)
        print('test_accuracy:%.2f%%' % (test_accuracy * 100))
        yy_test = [i for item in ytest for i in item]
        # print(yy_test)
        # print(len(yy_test))
        a = pd.DataFrame(preds.T)
        b = pd.DataFrame(yy_test)
        ab = pd.merge(a, b, right_index=True, left_index=True, suffixes=('', '_y'))
        ab.columns = ['preds', 'test']
        # 预测准确的投诉用户数
        com_r = ab.loc[(ab['preds'] == ab['test']) & ab['test'] == 1].count()
        # 真实投诉用户数
        com_count = ab.loc[ab['test'] == 1].count()
        # 预测投诉用户数
        pred_users = ab.loc[ab['preds'] == 1].count()
        # 查全率
        print("查全率:", com_r * 1.0 / com_count * 100)
        print("预测投诉用户数:", pred_users)
        # 计算查准率
        print("查准率:{}".format(com_r * 1.0 / pred_users * 100))

    def run(self):
        # 数据帅选
        traindata, test_X = self.choosedata()

        # 数据可视化特征
        self.fig_Picture(traindata)

        # 规范化各特征的值
        traindata.drop(['copm_count'], axis=1, inplace=True)
        traindata_1 = traindata.drop(['iscomp'], axis=1)
        traindata_1 = self.dataNormalized(traindata_1)

        # 用所有属性训练
        X = traindata_1.values
        b = traindata_1.columns
        y = traindata[['iscomp']].values
        # print("traindata1:",b)

        # 构建Xgboost模型，并使用GridSearchCV搜索最优参数的解
        # self.split_Train_TestData(X,y,b)

        print("所有属性训练模型结果：")
        importance_property, model1 = self.ModelTrain(X, y, b)

        # 用重要属性训练
        mm = list(importance_property)
        m = 'iscomp'
        mm.append(m)
        traindata_2 = traindata[mm]
        b2 = importance_property
        y2 = traindata_2[['iscomp']].values
        traindata_2 = traindata_2.drop(['iscomp'], axis=1)
        X2 = traindata_2.values

        print("重要属性训练模型结果：")
        # 选取重要属性训练模型
        importance_property2, model2 = self.ModelTrain(X2, y2, b2)
        # 模型存储路径
        path = 'D:\PycharmProjects\TSFL\poten_model'
        # 保存模型
        self.save_model(model2, path, 'xgb_predict.model')
        # 加载模型
        tar = self.load_model(path, 'xgb_predict.model')
        test_X[mm].to_csv('D:\临时文件/test_X1.csv', header=1, encoding='gbk')  # 保存列名存储
        # 模型预测
        test = test_X[mm].drop(['iscomp'], axis=1)
        y_test = test_X[['iscomp']].as_matrix()
        self.model_predict(tar, test, y_test)


if __name__ == "__main__":
    # 从数据库获取样本数据
    a = Postgresql()
    sql = "SELECT * FROM volte.vn_potential_complaint_result_100"
    data = a.GetData(sql)
    # print(data.columns)
    print(data.iloc[:, 0].size)
    data = pd.DataFrame(data)
    a.finish()
    data.to_csv('D:\临时文件/potentialdata.csv', header=1, encoding='gbk')
    # 调用potential类
    task = potential(data)
    task.run()
