# encoding=utf-8
import datetime
import psycopg2
import pandas as pd
import numpy as np
import time
from io import StringIO
import os
import csv

##读写日志
def write_csv(context):
    log = r'D:/集团工单/rizhi.csv'
    # log = r'/data/ftp/python/fcsv/log.csv'
    with open(log, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        r_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        data_row = [r_time, context]
        csv_write.writerow(data_row)
        f.close()

def connectionPosgresql():
    try:
        conn = psycopg2.connect(database="db", user="postgres_user", password="postgres_password", host="10.10.10.109",
                                port="5432")
        # conn = psycopg2.connect(database="db", user="cmdi_volte", password="Cmdi@2O19", host="192.169.5.142",port="35005")
    except Exception as e:
        print(e)
        context = e
        write_csv(context)
    return conn


##派单数据入postgresql库,参数a为要入的数，b入的表名
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
        context = "入数成功"
        print(context)
        write_csv(context)
    except Exception as e:
        context = e
        write_csv(context)
if __name__ == "__main__":
 ##回单数据入库
 ##派单数据入库
    eptable = pd.read_csv('D:/v_eptable702.csv', encoding='utf-8')
    eptable  = eptable.drop_duplicates('cgi')
    eptable['cgi'] = "460-00-" + eptable['cgi'].map(str)
    temp = pd.read_csv('D:/temp_0702.csv', encoding='utf-8')
    print(temp.iloc[:, 0].size)
    pm = pd.read_csv('D:/m_pm_cell.csv', encoding='gbk')
    pm['starttime'] = pm['starttime'].str[:9]
    pm = pm.groupby(['cgi','starttime'])['UPOCTUL', 'UPOCTDL'].agg(np.sum)
    pm = pm.reset_index(drop=False)
    print(pm.head())
    pm['UPOCTUL'] = pm['UPOCTUL']/1024
    pm['UPOCTDL'] = pm['UPOCTDL'] / 1024
    temp.rename(columns={'temp_wyp_cmdi_01_volte_request_cgi_count1.request_no': 'request_no'}, inplace=True)
    temp.rename(columns={'temp_wyp_cmdi_01_volte_request_cgi_count1.cgi': 'cgi'}, inplace=True)
    temp.rename(columns={'temp_wyp_cmdi_01_volte_request_cgi_count1.cell_from': 'cell_from'}, inplace=True)
    temp.rename(columns={'temp_wyp_cmdi_01_volte_request_cgi_count1.count1': 'count1'}, inplace=True)

    result_ep = pd.merge(temp,eptable,on = 'cgi',how='left',suffixes=('', '_y'))
    print(result_ep.iloc[:, 0].size)
    result = pd.merge(result_ep, pm, on='cgi', how='left', suffixes=('', '_y'))
    print(result.iloc[:, 0].size)
    # result.to_csv('D:/tousufenxi.csv', header=1, encoding='gbk', index=True)  # 保存列名存储
    writer = pd.ExcelWriter('D:/result.xlsx')
    result_ep.to_excel(writer, '投诉补充故障判断', index=True, encoding='gbk')
    result.to_excel(writer, '投诉补充流量判断', index=True, encoding='gbk')
    writer.save()

    # dateIntoPostgresql(send, 'volte.v_volte_send')
    # dateIntoPostgresql(v_return, 'volte.v_volte_returnvaluation')