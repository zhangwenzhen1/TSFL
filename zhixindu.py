from pyecharts import Line, Bar, Pie, EffectScatter
import pandas as pd
from pyecharts import Scatter
import numpy as np
from datetime import datetime

df = pd.read_csv('D:/vn_xdrresultscore1.csv', encoding='utf-8')
print(df.columns)
df = df[['request_no', 'accept_time', 'city', 'problem_sort', 'xdr_s1_end_time']]
df = df.loc[df['xdr_s1_end_time'].notnull()]
df['accept_time'] = pd.to_datetime(df['accept_time'], format='%Y/%m/%d %H:%M:%S')
df['xdr_s1_end_time'] = pd.to_datetime(df['xdr_s1_end_time'], format='%Y/%m/%d %H:%M:%S')
df['time_interval'] = df['accept_time'] - df['xdr_s1_end_time']
df['time_interval'] = pd.to_datetime(df['time_interval'], format='%Y/%m/%d %H:%M:%S')
df['hour'] = df['time_interval'].dt.hour
df['minute'] = df['time_interval'].dt.minute
df['second'] = df['time_interval'].dt.second
df['interval'] = round((df['hour'] * 3600 + df['minute'] * 60 + df['second']) * 1.0 / 300) + 1

print(df.head(100))

df1 = df.groupby(['interval'])['request_no'].count()
df1 = df1.reset_index(drop=False)
print(df1.head())

df1.rename(columns={'request_no': 'request_num', }, inplace=True)

df1.to_csv('D:\Volte\zhixindu.csv', encoding='gbk', index=False)
scatter = Scatter(title="话单匹配时间关系", width=6000, height=420)
scatter.add(name='', x_axis=df1['interval'], y_axis=df1['request_num'], yaxis_min=1, yaxis_max=200, xaxis_min=0,
            xaxis_max=300)
scatter.render(path='C:/Users\X1\Desktop/置信度散点图.html')
