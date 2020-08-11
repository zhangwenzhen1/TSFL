import numpy as np
import math
import pandas as pd
# df = pd.read_csv('D:\集团工单\eptable\eptable.csv', encoding='utf-8',nrows=10, usecols=['cityname', 'cgi', 'longitude', 'latitude'])
# print(df.iloc[:, 0].size)
# df['flag'] = 0
# df1 = df.head(5)
# print(df1.iloc[:, 0].size)
# print(df1.head(6))
# df = df.loc[(~df['cgi'].isin(df1['cgi']))]
# df.reset_index(drop=True,inplace=True)
# # # 构造输出结果表
# # lieming = ['cityname', 'cgi', 'longitude', 'latitude', '主小区cgi', 'lon', 'lat', '小区间距离']
# # Result = pd.DataFrame(columns=lieming)
# # print(len(Result['cgi']))
# print(df.head(10))
# print(df.loc[0, :])
df = pd.read_excel('D:\集团工单\无线指标2020-07-27.xlsx', sheet_name='无线指标2020-07-27',usecols=['地市', 'cgi', '频段'])
df = df.groupby(['地市','频段'])['cgi'].count()
df.to_csv('D:\集团工单\无线指标2020-07-27.csv', header=1, encoding='gbk', index=True)  # 保存列名存储
print(df.head())