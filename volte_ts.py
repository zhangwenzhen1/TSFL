import pandas as pd
import os
import numpy as np
##########################################################################################
path = r'D:\Volte'  #指定存放文件的地址
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
########################################################################################


df.rename(columns={'volte_merge_001': 'S-CSCF初始注册成功次数','volte_merge_002': 'S-CSCF初始注册请求次数',
                   'volte_merge_003': '主叫掉话次数','volte_merge_004': '被叫掉话次数',
                   'volte_merge_005': 'IMS初始注册请求次数','volte_merge_006': 'IMS初始注册成功次数',
                   'volte_merge_007': 'VOLTE语音呼叫总次数(MOC+MTC)','volte_merge_008': 'VOLTE语音网络呼叫接通次数(MOC+MTC)',
                   },inplace = True)

df['S-CSCF初始注册失败次数'] = df['S-CSCF初始注册请求次数']-df['S-CSCF初始注册成功次数']
df['IMS初始注册失败次数'] = df['IMS初始注册请求次数']-df['IMS初始注册成功次数']
df['VOLTE语音呼叫失败次数(MOC+MTC)'] =df['VOLTE语音呼叫总次数(MOC+MTC)'] - df['VOLTE语音网络呼叫接通次数(MOC+MTC)']
df['VoLTE语音异常次数'] = df['S-CSCF初始注册失败次数'] + df['主叫掉话次数'] +df['被叫掉话次数'] + df['IMS初始注册失败次数']+\
                    df['VOLTE语音呼叫失败次数(MOC+MTC)']
# print(df.head())

def f_sum(a):
    a = a.groupby(['msisdn'])['S-CSCF初始注册失败次数', '主叫掉话次数', '被叫掉话次数', 'IMS初始注册失败次数',
                              'VOLTE语音呼叫失败次数(MOC+MTC)', 'VoLTE语音异常次数'].agg(np.sum)
    a = a.reset_index(drop=False)
    return a
df = f_sum(df)

df1 = pd.read_csv('D:\Tsresulte\T_4.csv', encoding='gbk')
df1 =df1[['accept_msisdn_6']]
df1['是否投诉'] = 1
df1.rename(columns={'accept_msisdn_6': 'msisdn',},inplace = True)
# print(df['msisdn'])
df = pd.merge(df,df1,on='msisdn',how='left',suffixes=('', '_y')) # pandas csv表左连接
print(df.iloc[:,0].size)
df.to_csv('D:\Tsresulte\df.csv',header=1,encoding='gbk')
