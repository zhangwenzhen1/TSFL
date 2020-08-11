import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

df = pd.read_excel('D:\Volte\S1异常表整理.xlsx', encoding='gbk')
df = df[['temp2.msisdn','temp2.procedure_status']]
print(df.iloc[:,0].size)
# df = df.loc[((df['temp2.procedure_status'] == 1)  | (df['temp2.procedure_status'] == 255) | (df['temp2.procedure_status'] == 0)),
#             ['temp2.msisdn','temp2.procedure_status']]
# df = df.loc[((df['temp2.procedure_status'] == 1)  | (df['temp2.procedure_status'] == 255)),
#             ['temp2.msisdn','temp2.procedure_status']]
df = df.loc[( df['temp2.procedure_status'] == 0),['temp2.msisdn','temp2.procedure_status']]
print(df.iloc[:,0].size)
print(df.columns)
df2 = pd.read_csv('D:\Volte\T20200222.csv', encoding='gbk')
#########
df = df.loc[(df['temp2.msisdn'].isin(df2['accept_msisdn']))]
print(df.iloc[:,0].size)
df1 = df.groupby('temp2.msisdn').count()
# df1 = df1.to_frame()
df1.columns=['异常话单数']
df1 = df1.reset_index(drop=False)
print(df1.head(102))
df2 = df1.groupby('异常话单数').count()
df2.columns=['投诉用户数']
df2 = df2.reset_index(drop=False)
print(df2.columns)
# df1 = df.groupby(['temp2.msisdn'])['id'].count()
# df2.to_csv('D:\Volte\T20200217\Result\S1异常0.csv', encoding='gbk',index=False)
#绘图
plot1 = plt.plot(df2['异常话单数'], df2['投诉用户数'], 's',label='original values')
# plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('异常话单数')
plt.ylabel('投诉用户')
#把x轴的刻度间隔设置为1，并存在变量里
x_major_locator=MultipleLocator(1)

#把y轴的刻度间隔设置为10，并存在变量里
y_major_locator=MultipleLocator(1)

ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数

plt.xlim(0,90)
# plt.legend(loc=4) #指定legend的位置右下角
plt.title('投诉用户数与异常话单数关系')
plt.show()