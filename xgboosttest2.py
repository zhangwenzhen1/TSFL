# plot feature importance manually
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

data = pd.read_csv("D:\临时文件/data.csv")
Y = data[['iscomp']]

data.drop(['iscomp'], axis=1, inplace=True)
# 提取预测数据中所有特征名称
features = data.columns.values
# 规范化各特征的值
for feature in features:
    mean, std = data[feature].mean(), data[feature].std()
    data.loc[:, feature] = (data[feature] - mean) / std
print(data.columns)
# load data
dataset = load_iris()
# split data into X and y
X = dataset.data
print(X)
y = dataset.target
print(y)
# X = data.values
# print(X)
# print(Y)
# y = Y.values
# y = y[0]
# print(y[0])

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
# fit model no training data
# 使用GridSearchCV搜索最优参数的解
# model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax',num_class=3)
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
# model.fit(X_train, y_train)
gs.fit(X_train, y_train)
# 显示数据集的结果
print(gs.best_score_)
print(gs.best_params_)
'''
path = 'D:\PycharmProjects\TSFL\poten_model'
model.save_model(path + '/xgb.model')  # 用于存储训练出的模型
# feature importance
print(model.feature_importances_)
# 显示重要特征
plot_importance(model, importance_type='gain')
plt.show()
# make predictions for test data and evaluate
y_pred = model.predict(X_test)
print(y_pred)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:%.2f%%" % (accuracy * 100.0))

# fit model using each importance as a threshold
thresholds = np.sort(model.feature_importances_)

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    print(selection_model.feature_importances_)
    plot_importance(selection_model, importance_type='gain')
    plt.show()
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    # print(select_X_test.feature_importances_)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 5,
    'slient': 1,
    'eta': 0.1,
}

# load model and data in
tar = xgb.Booster(params=params,model_file= path + '/xgb.model')
dtest = xgb.DMatrix(X, label=y)
preds = tar.predict(dtest)
#
print(preds)
'''
