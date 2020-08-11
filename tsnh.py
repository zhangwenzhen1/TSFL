import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
'''
df = pd.read_csv('D:\Tsresulte\df.csv',encoding='gbk')
print(df.iloc[:,0].size)
df.drop(['VoLTE语音异常次数'], axis=1)
df = df.loc[(df['主叫掉话次数']>0) | (df['被叫掉话次数']>0) |(df['VOLTE语音呼叫失败次数(MOC+MTC)']>0)]
print(df.iloc[:,0].size)
df.rename(columns={'是否投诉':'投诉用户数',},inplace = True)
df['VoLTE语音异常次数'] = df['主叫掉话次数'] +df['被叫掉话次数'] +df['VOLTE语音呼叫失败次数(MOC+MTC)']

df['投诉用户数'].fillna(0,inplace = True)
df['异常用户数'] = 1
print(df.columns)


def f_sum(a,b,c,d):
    a = a.groupby([b])[c,d].agg(np.sum)
    a = a.reset_index(drop=False)
    return a

df1 = f_sum(df,'VoLTE语音异常次数','投诉用户数','异常用户数')
df1['用户投诉概率'] = df1['投诉用户数']*1./df1['异常用户数']
print(df1.columns)
#绘图
plot1 = plt.plot(df1['VoLTE语音异常次数'], df1['用户投诉概率'], 's',label='original values')
# plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('VoLTE语音异常次数')
plt.ylabel('用户投诉概率')
plt.xlim(0,50)
# plt.legend(loc=4) #指定legend的位置右下角
plt.title('curve_fit')
# plt.show()
'''
################################################
path = r'D:\Volte\day'  #指定存放文件的地址
# path = r'D:\Volte\T20200217\Temp'  #指定存放文件的地址
file_name_list = os.listdir(path)   #返回.csv格式所有文件名的列表list
#for循环获取所有.csv格式文件的绝对地址的列表list
file_dir_list=[os.path.join(path,x) for x in file_name_list]
print(file_dir_list)
print(len(file_dir_list))
#定义DataFrame类型的变量df用来存放获取的所有数据
df = pd.DataFrame()
j=0
#for循环逐个读取每个csv里面的数据
for i in file_name_list:
    if j in range(0,len(file_dir_list)):
        if (i[-3:] == 'csv'):  # 筛选只读取csv结尾的文件
            # read_csv方法，参数sheet_name表示读取的工作簿，skiprows表示忽略几行，usecols表示读取指定的列
            csv1 = pd.read_csv(file_dir_list[j])
            # concat方法合并多个文件的数据
            df = pd.concat([df, csv1], ignore_index=True)
            j=j+1
print(df.iloc[:,0].size)
###############################################################################################
# df = pd.read_csv('D:\Volte\Temp_20200131.csv',encoding='utf-8')
print(df.columns)
df = df[['temp2.accept_msisdn','temp2.comptype']]
df['temp2.comptype'] = df['temp2.comptype'].astype(str) #更改数据格式
df1 = df[df['temp2.comptype'].str.contains('接通')]
df2 = df[df['temp2.comptype'].str.contains('掉话')]
df = df1.append(df2)
# df.to_csv('D:\Volte\T20200217\Result\Result11.csv', encoding='gbk',index=False)
#####用户异常换单统计
df = df.groupby(['temp2.accept_msisdn'])['temp2.comptype'].count()
df = df.reset_index(drop=False)
# df = df.to_frame()
df.rename(columns={'temp2.accept_msisdn': '投诉用户','temp2.comptype': '异常话单数'},inplace = True)
print(df.iloc[:,0].size)
df['投诉用户'] = df['投诉用户'].astype(str) #更改数据格式
print(df.head(17))
#####异常换单用户数统计
df = df.groupby(['异常话单数'])['投诉用户'].count()
df = df.reset_index(drop=False)
print(df.head(100))
# df.to_csv('D:\Volte\T20200217\Result\Result1.csv', encoding='gbk',index=False)
#绘图
plot1 = plt.plot(df['异常话单数'], df['投诉用户'], 's',label='original values')
# plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
# plt.cbook('异常话单')
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
plt.title('31号至6号投诉用户数与1小时内异常话单数关系')
plt.show()

