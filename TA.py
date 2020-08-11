import pandas as pd
import numpy as np


z1 = pd.read_csv('D:\gxt\yfugai.csv', encoding='gbk')
z1 = z1.loc[z1['小区平均TA'].notnull()]
z1 = z1.drop_duplicates('CGI')
print(z1.iloc[:,0].size)
b = list(set(z1['小区平均TA']))
# print(type(b))
# print(len(b))
# print(b[2])
lieming = ['CGI', 'TA0', 'TA1', 'TA2', 'TA3', 'TA4', 'TA5', 'TA6', 'TA7', 'TA8','TA9', 'TA10', 'TA11', 'TA12', 'TA13',
           'TA14', 'TA15', 'TA16', 'TA17','TA18', 'TA19', 'TA20', 'TA21', 'TA22', 'TA23', 'TA24', 'TA25', 'TA26',
           'TA27', 'TA28', 'TA29', 'TA30', 'TA31', 'TA32', 'TA33', 'TA34', 'TA35','TA36', 'TA37', 'TA38', 'TA39',
           'TA40', 'TA41', 'TA42', 'TA43', 'TA44','时间提前量采样点总数', '最大采样点占比所在', '小区平均TA',
           '大于平均TA采样点总数','大于平均TA采样点占比']
# 构造输出结果表
Result = pd.DataFrame(columns=lieming)
n = 0
while n< len(b):
    temp = z1.loc[z1['小区平均TA']==b[n]]
    temp['大于平均TA采样点总数'] = temp.iloc[:,b[0]+2:46].apply(lambda x: x.sum(),axis=1)
    temp['大于平均TA采样点占比'] = temp['大于平均TA采样点总数']*1.0/temp['时间提前量采样点总数']
    # 把结果放入表Result中
    Result = Result.append(temp)
    n = n + 1

Result.to_csv('D:\gxt\Result1.csv',header=1,encoding='gbk',index=False)

