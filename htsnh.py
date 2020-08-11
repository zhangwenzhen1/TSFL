import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import csv_merge as C
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

################################################
path = r'D:\Volte\T20200217\day'  #指定存放文件的地址
df = C.csv_merge(path)
print(df.columns)
df.rename(columns={'day_id': 'hour_id'},inplace = True)
df.rename(columns={'volte_merge_001': 'S-CSCF初始注册成功次数','volte_merge_002': 'S-CSCF初始注册请求次数',
                   'volte_merge_003': '主叫掉话次数','volte_merge_004': '被叫掉话次数',
                   'volte_merge_005': 'IMS初始注册请求次数','volte_merge_006': 'IMS初始注册成功次数',
                   'volte_merge_007': 'VOLTE语音呼叫总次数(MOC+MTC)','volte_merge_008': 'VOLTE语音网络呼叫接通次数(MOC+MTC)',
                   },inplace = True)
df['未接通次数'] = df['VOLTE语音呼叫总次数(MOC+MTC)'] - df['VOLTE语音网络呼叫接通次数(MOC+MTC)']
df = df[['hour_id','msisdn','主叫掉话次数', '被叫掉话次数','未接通次数']]
df['异常话单次数'] = df['主叫掉话次数'] + df['被叫掉话次数'] + df['未接通次数']

print(df.columns)

def f_sum(a):
    a = a.groupby(['hour_id','msisdn'])[ '主叫掉话次数', '被叫掉话次数', '未接通次数','异常话单次数'].agg(np.sum)
    a = a.reset_index(drop=False)
    return a
df = f_sum(df)
df.rename(columns={'msisdn': '投诉用户'},inplace = True)
# df = df.loc[(df['异常话单次数']>1) & (df['异常话单次数']<9)]
print(df.head())

path2 =  r'D:\Volte\T20200217\Temp1'
df2 = C.csv_merge(path2)
df2 =df2[['temp2.accept_msisdn']]
df2.rename(columns={'temp2.accept_msisdn': '投诉用户'},inplace = True)
df2['是否投诉'] =1
print(df2.iloc[:,0].size)
df2 = df2.drop_duplicates('投诉用户',keep='first')
print(df2.iloc[:,0].size)
print(df2.head())
df['投诉用户'] = df['投诉用户'].astype(str) #更改数据格式
df['投诉用户']=df['投诉用户'].map(str.strip) #清空空格
df2['投诉用户'] = df2['投诉用户'].astype(str) #更改数据格式
df2['投诉用户']=df2['投诉用户'].map(str.strip) #清空空格
df =  pd.merge(df,df2,on = '投诉用户',how='left',suffixes=('', '_y'))
print(df.head())
df.to_csv('D:\Volte\T20200217\Result\Result3.csv', encoding='gbk',index=False)