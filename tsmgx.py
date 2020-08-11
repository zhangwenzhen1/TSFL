import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pyecharts import Line, Bar, Pie, EffectScatter
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

df = pd.read_csv('D:\Volte\V600.csv', encoding='gbk')
print(df.iloc[:,0].size)
df = df.loc[( df['slicetime'] >= 20191209) & ( df['slicetime'] < 20191216)]
# df = df.loc[( df['slicetime'] >= 20191201) & ( df['slicetime'] < 20200101)]

print(df.iloc[:,0].size)
df = df.drop_duplicates(['request_no','accept_msisdn'])
print(df.iloc[:,0].size)
print(df.columns)

df2 = df.groupby('accept_msisdn')['request_no'].count()
# df2.columns=['投诉单数']
df2 = df2.reset_index(drop=False)
df2.rename(columns={'request_no': '投诉单数'},inplace = True)
print(df2.head())

df3 = df2.groupby('投诉单数')['accept_msisdn'].count()
df3 = df3.reset_index(drop=False)
df3.rename(columns={'accept_msisdn': '投诉用户数'},inplace = True)
print(df3.head())

plot1 = plt.plot(df3['投诉单数'], df3['投诉用户数'], 's',label='original values')
# plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('投诉单数')
plt.ylabel('投诉用户数')
#把x轴的刻度间隔设置为1，并存在变量里
x_major_locator=MultipleLocator(1)
#
#把y轴的刻度间隔设置为10，并存在变量里
# y_major_locator=MultipleLocator(5)
#
ax=plt.gca()
# #ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# #把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数

# plt.ylim(1,5000)
# plt.xlim(1,35)
# plt.legend(loc=4) #指定legend的位置右下角
plt.title('(月)投诉用户数与投诉单数关系')
plt.show()

# 普通折线图
line = Line('折线图')
line.add('商家A', df3['投诉单数'], df3['投诉用户数'], mark_point=['max'],is_smooth=True)
# line.add('商家B', attr, v2, mark_point=['min'], is_smooth=True)
line.show_config()
line.render(path='D:\PycharmProjects\python/01-04折线图.html')