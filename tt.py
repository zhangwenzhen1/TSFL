import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('D:\Volte\T22.csv',encoding='utf-8')
print(df.columns)
df = df[['accept_msisdn','comptype']]
df['comptype'] = df['comptype'].astype(str) #更改数据格式
df1 = df[df['comptype'].str.contains('接通')]
df2 = df[df['comptype'].str.contains('掉话')]
df = df1.append(df2)
df['comptype']= df['comptype'].map(str.strip) #清空空格
df['accept_msisdn']=df['accept_msisdn'].astype('str') #更改数据格式
df['accept_msisdn']= df['accept_msisdn'].map(str.strip) #清空空格
df = df.drop_duplicates(['accept_msisdn'],keep='first')
print(df.columns)
df.to_csv('D:\Volte\T20200222.csv', encoding='gbk',index=False)