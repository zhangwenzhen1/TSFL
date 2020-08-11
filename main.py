# encoding=utf-8
from ModifyGDFormat import *
from dateutil.relativedelta import relativedelta
from JJtGd import *
from JJtGdFK import *
from Postgresql import *
import os
import warnings
warnings.filterwarnings('ignore')

def getFileName(path):
    file_name_list = os.listdir(path)  # 返回所有文件名的列表list
    print(file_name_list)
    a = pd.DataFrame(file_name_list)
    a.columns = ['文件名']
    a = a.sort_values(by=['文件名'], ascending=[False])  # 文件名降序排列
    filename = a.iloc[0, 0]  # 获取最新的文件名
    print('需处理的文件为:\n' + filename)
    c = input('请确认是否处理上面的文件，是输入y，否输入n\n')
    if c == 'y':
        print('程序开始执行')
        return filename
    else:
        print("您要读取的文件名错误或不存在，请修改文件名或上传文件到指定路径,再执行本程序")
        filename = input('请输入正确的文件名(带后缀)：')
        if filename in file_name_list:
            print('需处理的文件为:\n'+ filename +'\n文件正确，程序开始执行')
            return filename
        else:
            print("您要读取的文件不存在，请上传文件到指定路径,再执行本程序")

def main():
    path = r'D:/集团工单/gd'  # 指定存放文件的地址
    path2 = r'D:/集团工单'  # 指定存放结果文件
    # path = r'/data/ftp/python/xls'  # 指定存放文件的地址
    # path2 = r'/data/ftp/python/fcsv'  # 指定存放结果文件
    filename = getFileName(path)
    ###执行生成派单任务
    paidan = JJtGd(path, filename)
    df = paidan.Run()
    task = ModifyGDFormat(df)
    send, v_return, df_all = task.Run()
    print("集团下发派单数:{}".format(str(df_all.iloc[:, 0].size)))
    print("实际可派工单数:{}".format(str(send.iloc[:, 0].size)))
    send.to_csv(path2 + '/paidan1.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
    v_return.to_csv(path2 + '/回单1.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
    df_all.to_csv(path2 + '/pandan_all.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
    print("派单任务完成")
    # df.to_csv(path2 + '/df.csv', header=1, encoding='gbk', index=False)  # 保存列名存储

    ###获取反馈指标日期
    # df = pd.read_csv('D:\集团工单\gd/df.csv',encoding='gbk')
    f_date = df[['vcequestiontype', '日期']]
    f_date = f_date.drop_duplicates()
    f_date['starttime'] = pd.to_datetime(f_date['日期'], format='%Y/%m/%d').apply(lambda x: x - relativedelta(months=+1))
    f_date['gd_startdate'] = f_date['starttime'].astype(np.str)
    f_date['gd_startdate'] = f_date['gd_startdate'].apply([lambda x: x[:7]])
    f_date['gd_startdate'] = pd.to_datetime(f_date['gd_startdate'], format='%Y/%m')
    f_date['problemtype'] = np.where(f_date['vcequestiontype'].isin(['VOLTEJTD', 'ESRVCCQHC', 'VOLTERABDXG']), '周派单',
                                     '月派单')
    f_date.rename(columns={'starttime': 'data_date', }, inplace=True)
    f_date = f_date[['gd_startdate', 'problemtype', 'data_date']]
    f_date = f_date.drop_duplicates()
    print('反馈指标日期获取成功')
    f_date.to_csv(path2 + "/riqi1.csv", header=1, encoding='gbk', index=False)  # 保存列名存储'''
    ####执行反馈表任务
    fankui = JJtGdFK(path, filename)
    result = fankui.Go_run()
    date = df_all[['vcequestiontype', 'starttime', 'endtime']]
    date = date.drop_duplicates()
    date['starttime'] = pd.to_datetime(date['starttime'], format='%Y/%m/%d').apply(
        lambda x: x - relativedelta(months=+1))
    date['endtime'] = pd.to_datetime(date['endtime'], format='%Y/%m/%d').apply(lambda x: x - relativedelta(months=+1))
    date.rename(columns={'starttime': 'startdate', 'endtime': 'enddate', 'vcequestiontype': 'problemtype'},
                inplace=True)
    result = pd.merge(result, date, on='problemtype', how='left', suffixes=('', '_y'))  # pandas csv表左连接
    result = result[
        ['province', 'city', 'cgi', 'startdate', 'enddate', 'problemtype', 'voltetraval', 'volteradconnratio',
         'volteraddropratio', 'uppdcplossratio', 'downpdcplossratio', 'failoutgeran', 'srvcchosucratio',
         'uptzsamprate', 'downtzsamprate', 'flag']]
    ff = Postgresql()
    ff.dateIntoPostgresql(result, "volte.vn_gdcellkpi_group")
    print("反馈表数据入库完成")
    ff.dateIntoPostgresql(f_date, "volte.vn_gd_group_date")
    ff.finish()
    print("反馈指标日期入库完成")
    result.to_csv(path2 + "/back.csv", header=1, encoding='gbk', index=False)  # 保存列名存储
    print("任务完成")


if __name__ == "__main__":
    main()
