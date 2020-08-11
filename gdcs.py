# encoding=utf-8
import psycopg2
import pandas as pd
import time
import os
import csv
from dateutil.relativedelta import relativedelta
from io import StringIO
import datetime
import numpy as np


##筛选广东省问题小区
def chooseProvince(a):
    a = a.loc[a['省份'] == '广东']
    return a


def Date_Subtract_Month(sdata):
    data = (sdata - relativedelta(months=+1)).strftime("%Y-%m-%d %H:%M:%S")
    return data


def Get_gddata(date):
    date = date.drop_duplicates(['日期'], keep='first')
    date.loc[:, '日期'] = pd.to_datetime(date['日期'], format='%Y%m%d')
    date.loc[:, '日期'] = date.loc[:, '日期'].apply(lambda x: Date_Subtract_Month(x))
    return date


####读写日志
def write_csv(context):
    log = r'D:/集团工单/rizhi.csv'
    # log = r'/data/ftp/python/fcsv/log.csv'
    with open(log, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        r_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        data_row = [r_time, context]
        csv_write.writerow(data_row)
        f.close()


######获取工单生成日期
def getProblemDate(f):
    ######获取低接通，高掉话，切换差小区工单生成日期
    f['日期'] = pd.to_datetime(f['日期'], format='%Y%m%d')
    aa = f['日期'].max()
    aa = (aa + datetime.timedelta(days=1) - relativedelta(months=+1)).strftime("%Y-%m-%d %H:%M:%S")
    bb = f['日期'].min()
    bb = (bb - relativedelta(months=+1)).strftime("%Y-%m-%d %H:%M:%S")
    context = "获取工单生成日期成功"
    write_csv(context)
    return aa, bb


def getBackTablekpi(fpath):
    try:
        ###读取文件
        VOLTEJTD = pd.read_excel(fpath, sheet_name='低接入小区闭环管理反馈表', usecols=['省份', 'CGI', 'VoLTE话务量', 'VoLTE接通率'])
        VOLTERABDXG = pd.read_excel(fpath, sheet_name='高掉话小区闭环管理反馈表', usecols=['省份', 'CGI', '话务量', '掉话率'])
        ESRVCCQHC = pd.read_excel(fpath, sheet_name='低SRVCC无线切换成功率小区闭环管理反馈表',
                                  usecols=['省份', 'cgi', 'LTE到2G切换失败次数', 'SRVCC无线切换成功率'])
        VOLTESXGDB = pd.read_excel(fpath, sheet_name='上行高丢包小区闭环管理反馈表', usecols=['省份', 'CGI', '上行平均丢包率'])
        VOLTEXXGDB = pd.read_excel(fpath, sheet_name='下行高丢包小区闭环管理反馈表', usecols=['省份', 'CGI', '下行平均丢包率'])
        VOLTESXGTZ = pd.read_excel(fpath, sheet_name='上行高吞字小区反馈表', usecols=['省份', '小区cgi', '上行高吞字采样点占比'])
        VOLTEXXGTZ = pd.read_excel(fpath, sheet_name='下行高吞字小区反馈表', usecols=['省份', '小区cgi', '下行高吞字采样点占比'])

        #######
        df_VOLTEJTD = pd.read_excel(fpath, sheet_name='低接入小区明细参考表', usecols=['省份', '日期'])
        df_VOLTESXGDB = pd.read_excel(fpath, sheet_name='上行高丢包小区明细参考表', usecols=['省份', '日期'])
        df_VOLTEJTD = chooseProvince(df_VOLTEJTD)
        df_VOLTESXGDB = chooseProvince(df_VOLTESXGDB)
        DF = Get_gddata(df_VOLTEJTD)
        DF.loc[:, 'problemtype'] = '周派单'
        df1 = Get_gddata(df_VOLTESXGDB)
        df1.loc[:, 'problemtype'] = '月派单'
        riqi = DF.append(df1)
        riqi.rename(columns={'日期': 'data_date'}, inplace=True)

        aa, bb = getProblemDate(df_VOLTEJTD)
        riqi.loc[:, 'gd_startdate'] = bb
        print(riqi.head())
        cc, dd = getProblemDate(df_VOLTESXGDB)


        #####筛选广东省两高两低小区
        VOLTEJTD = chooseProvince(VOLTEJTD)
        VOLTERABDXG = chooseProvince(VOLTERABDXG)
        ESRVCCQHC = chooseProvince(ESRVCCQHC)
        VOLTESXGDB = chooseProvince(VOLTESXGDB)
        VOLTEXXGDB = chooseProvince(VOLTEXXGDB)
        VOLTESXGTZ = chooseProvince(VOLTESXGTZ)
        VOLTEXXGTZ = chooseProvince(VOLTEXXGTZ)

        VOLTEJTD.columns = ['province', 'cgi', 'voltetraval', 'volteradconnratio']
        VOLTEJTD['problemtype'] = 'VOLTEJTD'

        VOLTERABDXG.columns = ['province', 'cgi', 'voltetraval', 'volteraddropratio']
        VOLTERABDXG['problemtype'] = 'VOLTERABDXG'

        ESRVCCQHC.columns = ['province', 'cgi', 'failoutgeran', 'srvcchosucratio']
        ESRVCCQHC['problemtype'] = 'ESRVCCQHC'

        VOLTESXGDB.columns = ['province', 'cgi', 'uppdcplossratio']
        VOLTESXGDB['problemtype'] = 'VOLTESXGDB'

        VOLTEXXGDB.columns = ['province', 'cgi', 'downpdcplossratio']
        VOLTEXXGDB['problemtype'] = 'VOLTEXXGDB'

        VOLTESXGTZ.columns = ['province', 'cgi', 'uptzsamprate']
        VOLTESXGTZ['problemtype'] = 'VOLTESXGTZ'

        VOLTEXXGTZ.columns = ['province', 'cgi', 'downtzsamprate']
        VOLTEXXGTZ['problemtype'] = 'VOLTEXXGTZ'
        result = pd.concat([VOLTERABDXG, VOLTEJTD, ESRVCCQHC, VOLTESXGDB, VOLTEXXGDB, VOLTESXGTZ, VOLTEXXGTZ], axis=0,
                           ignore_index=True)
        result['startdate'] = np.where(result['problemtype'].isin(['VOLTEJTD', 'ESRVCCQHC', 'VOLTERABDXG']), bb, dd)
        result['enddate'] = np.where((result['problemtype'].isin(['VOLTEJTD', 'ESRVCCQHC', 'VOLTERABDXG'])), aa, cc)
        context = "集团“两高两低”反馈表KPI获取成功"
        write_csv(context)
        print(context)
    except Exception as e:
        context = e
        write_csv(context)
    return riqi, result


# fpath = r'D:/集团工单/gd/两低两高小区问题跟踪表20200301.xlsx'
#
# # df_VOLTEJTD = pd.read_excel(fpath, sheet_name='低接入小区明细参考表', usecols=['省份', '日期'])
# # df_VOLTESXGDB = pd.read_excel(fpath, sheet_name='上行高丢包小区明细参考表', usecols=['省份', '日期'])
# # df_VOLTEJTD = chooseProvince(df_VOLTEJTD)
# # df_VOLTESXGDB = chooseProvince(df_VOLTESXGDB)
#
# riqi, result = getBackTablekpi(fpath)
# # riqi.loc[:,'data_date'] = pd.to_datetime(riqi['data_date'], format='%Y%m')
# riqi['gd_startdate'] = riqi['gd_startdate'].astype(np.str)
# riqi['gd_startdate'] = riqi['gd_startdate'].apply([lambda x: x[:7]])
# riqi = riqi[['gd_startdate', 'problemtype', 'data_date']]
# riqi['gd_startdate'] = pd.to_datetime(riqi['gd_startdate'],format='%Y/%m/%d %H:%M:%S')
# print(riqi.head(20))
###连接数据库
def connectionPosgresql():
    try:
        conn = psycopg2.connect(database="db", user="postgres_user", password="postgres_password", host="10.10.10.109",
                                port="5432")
        # conn = psycopg2.connect(database="db", user="cmdi_volte", password="Cmdi@2O19", host="192.169.5.142",
        #                         port="35005")
    except Exception as e:
        print(e)
        context = e
        write_csv(context)
    return conn
#######csv数据入postgresql库,参数a为要入的数据，b入库的表名，c入库表的列名
def dateIntoPostgresql(a, b):
    try:
        col = a.columns
        # dataframe类型转换为IO缓冲区中的str类型
        output = StringIO()
        a.to_csv(output, sep='\t', index=False, header=False)
        output1 = output.getvalue()
        conn = connectionPosgresql()
        cur = conn.cursor()
        cur.copy_from(StringIO(output1), b, null='', columns=col)
        conn.commit()
        cur.close()
        conn.close()
        context = "集团“两高两低”反馈表入数成功"
        print(context)
        write_csv(context)
    except Exception as e:
        context = e
        write_csv(context)
        print('入数报错')
riqi = pd.read_csv('D:\集团工单/riqi.csv', encoding='gbk')

dateIntoPostgresql(riqi, "volte.vn_gd_group_date")