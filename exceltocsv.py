import pandas as pd
import numpy as np
'''
#Excel转CSV，一个sheet表存入一个CSV文件中，并以sheet名字命名
def excel_to_csv_pd():
    #sheet_name=None表示读取全部sheet，或者sheet_name=[0,10]，此处用sheet_name，而不是用sheetname
    data_xls = pd.read_excel('C:/Users\Administrator\Desktop/test/1.xlsx',sheet_name=None,index_col=0)
    for key in data_xls:
        data_xls[key].to_csv('C:/Users\Administrator\Desktop/test\’+key+’.csv', encoding='utf-8')


if __name__ == '__main__':
    excel_to_csv_pd()

#csv转Excel
import pandas as pd

def csv_to_xlsx_pd():
    csv = pd.read_csv('C:/Users\Administrator\Desktop/test/1.csv', encoding='utf-8')
    csv.to_excel('C:/Users\Administrator\Desktop/test/2.xlsx', sheet_name='data')
if __name__ == '__main__':
    csv_to_xlsx_pd()


def read1_1(fpath, num):
    for i in range(num):
        io = pd.io.excel.ExcelFile(fpath)
        data = pd.read_excel(io, sheetname='持仓明细')
        data2 = pd.read_excel(io, sheetname='成交明细')
        data2 = pd.read_excel(io, sheetname='成交明细')
        # data =pd.read_excel(io, sheetname=4)
        io.close()


def read1_2(fpath, num):
    for i in range(num):
        io = pd.io.excel.ExcelFile(fpath)
        # data =pd.read_excel(io, sheetname='持仓明细')
        data = pd.read_excel(io, sheetname=4)
        data2 = pd.read_excel(io, sheetname=2)
        data2 = pd.read_excel(io, sheetname=2)
        io.close()

'''


def f(df):
    df['new_rank'] = range(1, len(df) + 1)
    return df

# fpath ='C:/Users\X1\Desktop/VIP小区收集.xlsx'
# io = pd.io.excel.ExcelFile(fpath)

# sheet_name=None表示读取全部sheet，或者sheet_name=[0,10]，此处用sheet_name，而不是用sheetname
df = pd.read_excel('C:/Users\X1\Desktop/VIP小区收集.xlsx', sheet_name='业务维度')

# df = pd.read_excel(io, sheetname='业务维度')

# data = df.sort_values(['流量'], ascending=[ True, False]).groupby(['所属地市']).cumcount() + 1

df['Throughput_sort']= df['流量'].groupby(df['所属地市']).rank(ascending=0,method='dense')
# data = df.groupby('所属地市').sort_values(by=['流量'], ascending=(True, False)).apply(f) # 200多秒（100多万行数据集上）

# print(df.head())
df = df.sort_values(by=['所属地市','Throughput_sort'],ascending=[True,False])
df1 = df.groupby(['所属地市']).head(1)
df1['前30%序号'] = df['Throughput_sort'] * 0.3
df1 = df1[['所属地市','前30%序号']]
df1 = df1.groupby('所属地市')['前30%序号'].agg(np.mean)
df['前30%序号'] = df['所属地市'].map(df1)
print(df.head(42))


# df1 =df.head(-1)

# df.to_csv('C:/Users\X1\Desktop/VIP小区收集.csv', encoding='gbk',index=False)