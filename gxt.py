import pandas as pd
import numpy as np
writer = pd.ExcelWriter('D:\gxt/result.xlsx')
writer_1 = pd.ExcelWriter('D:\GDresult\gdcountresult.xlsx')
# df = pd.read_excel('C:/Users\X1\Desktop/VIP小区收集.xlsx', sheet_name='业务维度')
#
# ##对各地市流量分组排序并标出序号
# df['Throughput_sort']= df['流量'].groupby(df['所属地市']).rank(ascending=0,method='dense')
#
# # 按地市序号降序排列
# df = df.sort_values(by=['所属地市','Throughput_sort'],ascending=[True,False])
# # 取每个地市的最大序号
# df1 = df.groupby(['所属地市']).head(1)
# #计算排名第30%的序号
# df1['前30%序号'] = df['Throughput_sort'] * 0.3
#
# df1 = df1[['所属地市','前30%序号']]
# df1 = df1.groupby('所属地市')['前30%序号'].agg(np.mean)
# #序号并入排序后的表
# df['前30%序号'] = df['所属地市'].map(df1)
#
# df = df.loc[(df['Throughput_sort'] <= df['前30%序号'])]
#
# df.to_csv('C:/Users\X1\Desktop/VIP小区收集1.csv', encoding='gbk',index=False)