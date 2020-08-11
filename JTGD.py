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
        conn = psycopg2.connect(database="**", user="**", password="**", host="**",port="**")
    except Exception as e:
        print(e)
        context = e
        write_csv(context)
    return conn

##筛选广东省为题小区
def chooseProvince(a):
    a = a.loc[a['省份'] == '广东']
    return a
"""
##获取文件名
def getFileName(path):
    try:
        file_name_list = os.listdir(path)  # 返回所有文件名的列表list
        a = pd.DataFrame(file_name_list)
        a.columns = ['文件名']
        a = a.sort_values(by=['文件名'], ascending=[False])  ##文件名降序排列
        b = a.iloc[0, 0]  ####获取最新的文件名
        context = "获取文件名成功" + b
        write_csv(context)
        print(context)
    except Exception as e:
        context = e
        write_csv(context)
    return b
"""
##获取文件名
def getFileName(path):
    try:
        file_name_list = os.listdir(path)  # 返回所有文件名的列表list
        print(file_name_list)
        while True:
            filename = input("请输入您要读取的文件名：")
            if filename in file_name_list:
                context = "获取文件名成功:" + filename
                write_csv(context)
                print(context)
                return filename
            else:
                print("您要读取的文件不存在，请上传文件到指定路径")
    except Exception as e:
        context = e
        write_csv(context)

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

##获取集团“两高两低”小区
def getProblemCellKpi(fpath):
    try:
        ##读取文件
        df_VOLTEJTD = pd.read_excel(fpath, sheet_name='低接入小区明细参考表')
        df_VOLTERABDXG = pd.read_excel(fpath, sheet_name='高掉话小区明细参考表')
        df_ESRVCCQHC = pd.read_excel(fpath, sheet_name='低SRVCC无线切换成功率小区明细参考表')
        df_VOLTESXGDB = pd.read_excel(fpath, sheet_name='上行高丢包小区明细参考表')
        df_VOLTEXXGDB = pd.read_excel(fpath, sheet_name='下行高丢包小区明细参考表')
        df_VOLTESXGTZ = pd.read_excel(fpath, sheet_name='上行高吞字小区明细参考表')
        df_VOLTEXXGTZ = pd.read_excel(fpath, sheet_name='下行高吞字小区明细参考表')

        ##修改列名
        df_VOLTEJTD.rename(columns={'小区名称': 'vcauxiliarypointer4'}, inplace=True)
        df_VOLTERABDXG.rename(columns={'小区名称': 'vcauxiliarypointer4'}, inplace=True)
        df_ESRVCCQHC.rename(columns={'小区名称': 'vcauxiliarypointer4'}, inplace=True)
        df_VOLTESXGDB.rename(columns={'小区名称': 'vcauxiliarypointer4'}, inplace=True)
        df_VOLTEXXGDB.rename(columns={'小区名称': 'vcauxiliarypointer4'}, inplace=True)
        df_VOLTESXGTZ.rename(columns={'小区cgi': 'CGI', '小区名称': 'vcauxiliarypointer4'}, inplace=True)
        df_VOLTEXXGTZ.rename(columns={'小区cgi': 'CGI', '小区名称': 'vcauxiliarypointer4'}, inplace=True)

        ##筛选广东省两高两低小区
        df_VOLTEJTD = chooseProvince(df_VOLTEJTD)
        df_ESRVCCQHC = chooseProvince(df_ESRVCCQHC)
        df_VOLTERABDXG = chooseProvince(df_VOLTERABDXG)
        df_VOLTESXGDB = chooseProvince(df_VOLTESXGDB)
        df_VOLTEXXGDB = chooseProvince(df_VOLTEXXGDB)
        df_VOLTESXGTZ = chooseProvince(df_VOLTESXGTZ)
        df_VOLTEXXGTZ = chooseProvince(df_VOLTEXXGTZ)

        aa, bb = getProblemDate(df_VOLTEJTD)
        cc, dd = getProblemDate(df_VOLTESXGDB)

        ##获取最差指标小区信息
        df_VOLTEJTD = df_VOLTEJTD.sort_values(by=['CGI', '无线接通率(QCI1)'], ascending=[True, True])
        df_VOLTEJTD = df_VOLTEJTD.drop_duplicates(['CGI'], keep='first')
        df_VOLTEJTD['vccquestiontype'] = 'VoLTE无线接通差小区(语音)-19'
        df_VOLTEJTD['vcequestiontype'] = 'VOLTEJTD'
        df_VOLTEJTD['无线接通率(QCI1)'] = df_VOLTEJTD['无线接通率(QCI1)'].apply(lambda x: format(x, '.2%'))
        df_VOLTEJTD['vcproblemtarget1'] = "[无线接通率(QCI1):]" + df_VOLTEJTD['无线接通率(QCI1)'] + " [VoLTE话务量]:" + \
                                                 df_VOLTEJTD['VoLTE语音话务量'].map(str)
        df_VOLTEJTD = df_VOLTEJTD[
            ['vccquestiontype', 'vcequestiontype', 'CGI', 'vcproblemtarget1', 'vcauxiliarypointer4']]

        df_ESRVCCQHC = df_ESRVCCQHC.sort_values(by=['CGI', 'SRVCC无线切换成功率'], ascending=[True, True])
        df_ESRVCCQHC = df_ESRVCCQHC.drop_duplicates(['CGI'], keep='first')
        df_ESRVCCQHC['vccquestiontype'] = 'eSRVCC切换差小区-19'
        df_ESRVCCQHC['vcequestiontype'] = 'ESRVCCQHC'
        df_ESRVCCQHC['SRVCC无线切换成功率'] = df_ESRVCCQHC['SRVCC无线切换成功率'].apply(lambda x: format(x, '.2%'))
        df_ESRVCCQHC['vcproblemtarget1'] = "[eSRVCC切换成功率]" + df_ESRVCCQHC['SRVCC无线切换成功率'] + "[eSRVCC切换失败次数]" \
                                                  + df_ESRVCCQHC['SRVCC无线切换失败次数'].map(str)
        df_ESRVCCQHC = df_ESRVCCQHC[
            ['vccquestiontype', 'vcequestiontype', 'CGI', 'vcproblemtarget1', 'vcauxiliarypointer4']]

        df_VOLTERABDXG = df_VOLTERABDXG.sort_values(by=['CGI', 'E-RAB掉线率(QCI1)(小区级)'], ascending=[True, False])
        df_VOLTERABDXG = df_VOLTERABDXG.drop_duplicates(['CGI'], keep='first')
        df_VOLTERABDXG['vccquestiontype'] = 'VoLTE E-RAB掉线高小区(语音)-19'
        df_VOLTERABDXG['vcequestiontype'] = 'VOLTERABDXG'
        df_VOLTERABDXG['E-RAB掉线率(QCI1)(小区级)'] = df_VOLTERABDXG['E-RAB掉线率(QCI1)(小区级)'].apply(
            lambda x: format(x, '.2%'))
        df_VOLTERABDXG['vcproblemtarget1'] = "[E-RAB掉线率(QCI1)(小区级)]:" + df_VOLTERABDXG['E-RAB掉线率(QCI1)(小区级)'] + \
                                                    "[VoLTE话务量]:" + df_VOLTERABDXG['VoLTE语音话务量'].map(str)
        df_VOLTERABDXG = df_VOLTERABDXG[
            ['vccquestiontype', 'vcequestiontype', 'CGI', 'vcproblemtarget1', 'vcauxiliarypointer4']]

        df_VOLTESXGDB = df_VOLTESXGDB.sort_values(by=['CGI', '上行平均丢包率'], ascending=[True, False])
        df_VOLTESXGDB = df_VOLTESXGDB.drop_duplicates(['CGI'], keep='first')
        df_VOLTESXGDB['vccquestiontype'] = 'VoLTE上行高丢包小区-19'
        df_VOLTESXGDB['vcequestiontype'] = 'VOLTESXGDB'
        df_VOLTESXGDB['上行平均丢包率'] = df_VOLTESXGDB['上行平均丢包率'].apply(lambda x: format(x, '.2%'))
        df_VOLTESXGDB['vcproblemtarget1'] = "[上行丢包率]:" + df_VOLTESXGDB['上行平均丢包率'] + "[上行丢包率采样点点数]:" + \
                                                   df_VOLTESXGDB['上行总样本数'].map(str)
        df_VOLTESXGDB = df_VOLTESXGDB[
            ['vccquestiontype', 'vcequestiontype', 'CGI', 'vcproblemtarget1', 'vcauxiliarypointer4']]

        df_VOLTEXXGDB = df_VOLTEXXGDB.sort_values(by=['CGI', '下行平均丢包率'], ascending=[True, False])
        df_VOLTEXXGDB = df_VOLTEXXGDB.drop_duplicates(['CGI'], keep='first')
        df_VOLTEXXGDB['vccquestiontype'] = 'VoLTE下行高丢包小区-19'
        df_VOLTEXXGDB['vcequestiontype'] = 'VOLTEXXGDB'
        df_VOLTEXXGDB['下行平均丢包率'] = df_VOLTEXXGDB['下行平均丢包率'].apply(lambda x: format(x, '.2%'))
        df_VOLTEXXGDB['vcproblemtarget1'] = "[下行丢包率]:" + df_VOLTEXXGDB['下行平均丢包率'] + "[下行丢包率采样点点数]:" + \
                                                   df_VOLTEXXGDB['下行总样本数'].map(str)
        df_VOLTEXXGDB = df_VOLTEXXGDB[
            ['vccquestiontype', 'vcequestiontype', 'CGI', 'vcproblemtarget1', 'vcauxiliarypointer4']]

        df_VOLTESXGTZ = df_VOLTESXGTZ.sort_values(by=['CGI', '上行高吞字采样点占比'], ascending=[True, False])
        df_VOLTESXGTZ = df_VOLTESXGTZ.drop_duplicates(['CGI'], keep='first')
        df_VOLTESXGTZ['vccquestiontype'] = 'VoLTE上行高吞字小区-19'
        df_VOLTESXGTZ['vcequestiontype'] = 'VOLTESXGTZ'
        df_VOLTESXGTZ['上行高吞字采样点占比'] = df_VOLTESXGTZ['上行高吞字采样点占比'].apply(lambda x: format(x, '.2%'))
        df_VOLTESXGTZ['vcproblemtarget1'] = "[上行高吞字率]:" + df_VOLTESXGTZ['上行高吞字采样点占比'] + "[上行丢包率采样点点数]:" + \
                                                   df_VOLTESXGTZ['上行总采样点数'].map(str)
        df_VOLTESXGTZ = df_VOLTESXGTZ[
            ['vccquestiontype', 'vcequestiontype', 'CGI', 'vcproblemtarget1', 'vcauxiliarypointer4']]

        df_VOLTEXXGTZ = df_VOLTEXXGTZ.sort_values(by=['CGI', '下行高吞字采样点占比'], ascending=[True, False])
        df_VOLTEXXGTZ = df_VOLTEXXGTZ.drop_duplicates(['CGI'], keep='first')
        df_VOLTEXXGTZ['vccquestiontype'] = 'VoLTE下行高吞字小区-19'
        df_VOLTEXXGTZ['vcequestiontype'] = 'VOLTEXXGTZ'
        df_VOLTEXXGTZ['下行高吞字采样点占比'] = df_VOLTEXXGTZ['下行高吞字采样点占比'].apply(lambda x: format(x, '.2%'))
        df_VOLTEXXGTZ['vcproblemtarget1'] = "[下行高吞字率]:" + df_VOLTEXXGTZ['下行高吞字采样点占比'] + "[下行丢包率采样点点数]:" + \
                                                   df_VOLTEXXGTZ['下行总采样点数'].map(str)
        df_VOLTEXXGTZ = df_VOLTEXXGTZ[
            ['vccquestiontype', 'vcequestiontype', 'CGI', 'vcproblemtarget1', 'vcauxiliarypointer4']]
        df = df_VOLTEJTD.append(df_ESRVCCQHC).append(df_VOLTERABDXG).append(df_VOLTESXGDB).append(df_VOLTEXXGDB).append(df_VOLTESXGTZ).append(df_VOLTEXXGTZ)
        df = df.reset_index(drop=False)
        context = "集团两高两低小区kpi获取成功"
        write_csv(context)
    except Exception as e:
        context = e
        write_csv(context)
    return df, aa, bb, cc, dd

##获取工单生成日期
def getProblemDate(a):
    ##获取低接通，高掉话，切换差小区工单生成日期
    a['日期'] = pd.to_datetime(a['日期'], format='%Y%m%d')
    aa = a['日期'].max()
    aa = (aa + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    bb = a['日期'].min()
    context = "获取工单生成日期成功"
    write_csv(context)
    return aa, bb

##连接数据库获取上次派单最大编号及最新工参数据表
def getPublicParameter():
    conn = connectionPosgresql()
    cur = conn.cursor()
    cur.execute("SELECT vcauxiliarypointer6 FROM volte.v_volte_send where vcauxiliarypointer6 like 'JT%'")
    rows = cur.fetchall()
    t = pd.DataFrame(rows)
    t.columns = ['vcauxiliarypointer6']
    t['vcauxiliarypointer6'] = t['vcauxiliarypointer6'].apply(lambda x: x.replace('JT', ''))
    t['vcauxiliarypointer6'] = t['vcauxiliarypointer6'].astype(np.int)
    suoyin = t['vcauxiliarypointer6'].max()
    # print(suoyin)
    cur.execute("SELECT cgi,state,cityname,districtandcounty,vendor FROM volte.v_eptable ")
    eptable = cur.fetchall()
    eptable = pd.DataFrame(eptable)
    conn.commit()
    cur.close()
    conn.close()
    context = "获取上次派单最大编号及工参成功"
    write_csv(context)
    return suoyin, eptable

##筛选现网有业务工单
def chooseState(df, eptable):
    df_all = pd.merge(df, eptable, on='CGI', how='left', suffixes=('', '_y'))  # pandas csv表左连接
    df = df_all.loc[df_all['state'] == '现网有业务']
    df.drop(columns=['state'], axis=1, inplace=True)
    context = "获取现网有业务工单成功"
    write_csv(context)
    return df, df_all

##修改为省派单式
def modifyGDFormat(df):
    df['vcquestioncategory'] = "小区问题点"
    df['vcnetworksystem'] = "LTE"
    df['vcdatasource'] = "端到端信令分析优化"
    df['vccgi'] = "460-00-" + df['CGI'].map(str)
    df['vcspecialtag'] = "VOLTE两高两低-集团"
    df['vcproblemtarget3'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    df['vcproblemtarget5'] = "广东"
    df['vcproblemtarget2'] = "VOLTE-JTGD" + '-' + (
        time.strftime('%Y%m%d', time.localtime(time.time()))).format_map(str) \
                                    + '-' + df['vcauxiliarypointer6']
    send = df[['vcquestioncategory', 'vccquestiontype', 'vcequestiontype', 'vcnetworksystem', 'starttime', 'endtime',
               'vcdatasource', 'vccgi', 'vcspecialtag', 'vcproblemtarget1', 'vcproblemtarget2', 'vcproblemtarget3',
               'vcproblemtarget5', 'vcauxiliarypointer1', 'vcauxiliarypointer2', 'vcauxiliarypointer3',
               'vcauxiliarypointer4', 'vcauxiliarypointer6']]
    lieming = ['vcequestiontype', 'vcnetworksystem', 'starttime', 'endtime', 'vcauxiliarypointer6']
    v_return = pd.DataFrame(columns=lieming)
    v_return['vcequestiontype'] = df['vcequestiontype']
    v_return['vcnetworksystem'] = df['vcnetworksystem']
    v_return['starttime'] = df['starttime']
    v_return['endtime'] = df['endtime']
    v_return['vcauxiliarypointer6'] = df['vcauxiliarypointer6']
    v_return = v_return[['vcequestiontype', 'vcnetworksystem', 'starttime', 'endtime', 'vcauxiliarypointer6']]
    context = "修改工单格式成功"
    write_csv(context)
    return send, v_return

if __name__ == "__main__":
    path = r'D:/集团工单/gd'  # 指定存放文件的地址
    path2 = r'D:/集团工单'  # 指定存放文件的地址
    # path = r'/data/ftp/python/xls'  # 指定存放文件的地址
    # path2 = r'/data/ftp/python/fcsv'  # 指定存放结果文件
    ##获取目标文件名
    b = getFileName(path)
    ##获取目标文件路径
    fpath = path + '/' + b
    print("文件路径获取成功")
    ##获取集团“两高两低”小区
    df, aa, bb, cc, dd = getProblemCellKpi(fpath)
    print("集团“两高两低”小区获取成功")
    ##获取上次派单最大编号，及工参表
    suoyin, eptable = getPublicParameter()
    eptable.columns = ['CGI', 'state', 'vcauxiliarypointer1', 'vcauxiliarypointer2', 'vcauxiliarypointer3']
    eptable = eptable.drop_duplicates('CGI')

    df['starttime'] = np.where(df['vcequestiontype'].isin(['VOLTEJTD', 'ESRVCCQHC', 'VOLTERABDXG']), bb, dd)
    df['endtime'] = np.where((df['vcequestiontype'].isin(['VOLTEJTD', 'ESRVCCQHC', 'VOLTERABDXG'])), aa, cc)
    ##筛选现网有业务工单
    df, df_all = chooseState(df, eptable)
    df = df.reset_index(drop=False)
    df['vcauxiliarypointer6'] = "JT" + (df.index + 1 + suoyin).map(str)
    df_all = df_all[['vccquestiontype', 'vcequestiontype', 'starttime', 'endtime', 'CGI', 'vcauxiliarypointer4',
                     'vcauxiliarypointer1', 'vcauxiliarypointer2', 'vcauxiliarypointer3', 'state', 'vcproblemtarget1']]
    df_all.columns = ['vccquestiontype', 'vcequestiontype', 'starttime', 'endtime', 'cgi', 'cellname', 'city',
                      'districtandcounty', 'vendor', 'state', 'vcproblemtarget']
    ##修改为省派单式
    send, v_return = modifyGDFormat(df)
    send.loc[:,'vcauxiliarypointer1'] = np.where(
        (send['vcauxiliarypointer1'].isnull() | (send['vcauxiliarypointer1'] == '')), '未识别地市',send['vcauxiliarypointer1'])

    print("集团下发派单数:{}".format(str(df_all.iloc[:, 0].size)))
    print("实际可派工单数:{}".format(str(send.iloc[:, 0].size)))
    send.to_csv(path2 + '/send1.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
    v_return.to_csv(path2 + '/return1.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
    df_all.to_csv(path2 + '/send_all.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
    ##派单数据入库
    dateIntoPostgresql(send, 'volte.v_volte_send')
    context1 = "派单表数据入库成功"
    write_csv(context1)
    print(context1)
    ##回单数据入库
    dateIntoPostgresql(v_return, 'volte.v_volte_returnvaluation')
    context2 = "回单表数据入库成功"
    write_csv(context2)
    print(context2)
    ##集团非现网有业务工单数据入库
    dateIntoPostgresql(df_all, 'volte.vn_volte_gdsend_jt')
    context3 = "集团所有工单数据入库成功"
    write_csv(context3)
    print(context3)
    print("任务执行完成")
