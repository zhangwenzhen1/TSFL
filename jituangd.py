import pandas as pd
import os
import numpy as np
import time
import datetime
def function(a):
    a = a.loc[a['省份']=='广东']
    return a

##获取上次集团派单的最大编号
bianhao_s = pd.read_csv('D:/集团工单/bianhao.csv',encoding='gbk')
suoyin = bianhao_s.iloc[0,0]

######文件路径
fpath = r'D:/集团工单/附件6：两低两高小区问题跟踪表(2020年3月).xlsx'  #指定存放文件的地址f

###读取文件
df_VOLTEJTD = pd.read_excel(fpath, sheet_name='低接入小区明细参考表')
df_VOLTERABDXG = pd.read_excel(fpath, sheet_name='高掉话小区明细参考表')
df_ESRVCCQHC = pd.read_excel(fpath, sheet_name='低SRVCC无线切换成功率小区明细参考表')
df_VOLTESXGDB = pd.read_excel(fpath, sheet_name='上行高丢包小区明细参考表')
df_VOLTEXXGDB = pd.read_excel(fpath, sheet_name='下行高丢包小区明细参考表')
df_VOLTESXGTZ = pd.read_excel(fpath, sheet_name='上行高吞字小区明细参考表')
df_VOLTEXXGTZ = pd.read_excel(fpath, sheet_name='下行高吞字小区明细参考表')

###修改列名
df_VOLTEJTD.rename(columns={'小区名称': 'vcauxiliarypointer4'},inplace = True)
df_VOLTERABDXG.rename(columns={'小区名称': 'vcauxiliarypointer4'},inplace = True)
df_ESRVCCQHC.rename(columns={'小区名称': 'vcauxiliarypointer4'},inplace = True)
df_VOLTESXGDB.rename(columns={'小区名称': 'vcauxiliarypointer4'},inplace = True)
df_VOLTEXXGDB.rename(columns={'小区名称': 'vcauxiliarypointer4'},inplace = True)
df_VOLTESXGTZ.rename(columns={'小区cgi': 'CGI','小区名称': 'vcauxiliarypointer4'},inplace = True)
df_VOLTEXXGTZ.rename(columns={'小区cgi': 'CGI','小区名称': 'vcauxiliarypointer4'},inplace = True)

#####筛选广东省两高两低小区
df_VOLTEJTD = function(df_VOLTEJTD)
df_ESRVCCQHC = function(df_ESRVCCQHC)
df_VOLTERABDXG = function(df_VOLTERABDXG)
df_VOLTESXGDB = function(df_VOLTESXGDB)
df_VOLTEXXGDB = function(df_VOLTEXXGDB)
df_VOLTESXGTZ = function(df_VOLTESXGTZ)
df_VOLTEXXGTZ = function(df_VOLTEXXGTZ)

######获取低接通，高掉话，切换差小区工单生成日期
df_VOLTEJTD['日期'] = pd.to_datetime(df_VOLTEJTD['日期'],format='%Y%m%d')
aa = df_VOLTEJTD['日期'].max()
aa = (aa+datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
bb = df_VOLTEJTD['日期'].min()

######获取高呑子，高丢包小区工单生成日期
df_VOLTESXGDB['日期'] = pd.to_datetime(df_VOLTESXGDB['日期'],format='%Y%m%d')
cc = df_VOLTESXGDB['日期'].max()
cc = (cc+datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
dd = df_VOLTESXGDB['日期'].min()

#######获取最差指标小区信息
df_VOLTEJTD = df_VOLTEJTD.sort_values(by=['CGI','无线接通率(QCI1)'],ascending=[True,True])
df_VOLTEJTD = df_VOLTEJTD.drop_duplicates(['CGI'],keep='first')

df_VOLTEJTD['vccquestiontype'] = 'VoLTE无线接通差小区(语音)-20'
df_VOLTEJTD['vcequestiontype'] = 'VOLTEJTD'
df_VOLTEJTD['无线接通率(QCI1)'] = df_VOLTEJTD['无线接通率(QCI1)'].apply(lambda x: format(x, '.2%'))
df_VOLTEJTD['vcproblemtarget1'] = "[无线接通率(QCI1):]" + df_VOLTEJTD['无线接通率(QCI1)'] +" [VoLTE话务量]:" +\
                                  df_VOLTEJTD['VoLTE语音话务量'].map(str)
df_VOLTEJTD = df_VOLTEJTD[['vccquestiontype','vcequestiontype','CGI','vcproblemtarget1','vcauxiliarypointer4']]

df_ESRVCCQHC = df_ESRVCCQHC.sort_values(by=['CGI','SRVCC无线切换成功率'],ascending=[True,True])
df_ESRVCCQHC = df_ESRVCCQHC.drop_duplicates(['CGI'],keep='first')

df_ESRVCCQHC['vccquestiontype'] = 'eSRVCC切换差小区-20'
df_ESRVCCQHC['vcequestiontype'] = 'ESRVCCQHC'
df_ESRVCCQHC['SRVCC无线切换成功率']= df_ESRVCCQHC['SRVCC无线切换成功率'].apply(lambda x: format(x, '.2%'))
df_ESRVCCQHC['vcproblemtarget1'] = "[eSRVCC切换成功率]" + df_ESRVCCQHC['SRVCC无线切换成功率']+"[eSRVCC切换失败次数]" \
                                  + df_ESRVCCQHC['SRVCC无线切换失败次数'].map(str)
df_ESRVCCQHC = df_ESRVCCQHC[['vccquestiontype','vcequestiontype','CGI','vcproblemtarget1','vcauxiliarypointer4']]

df_VOLTERABDXG = df_VOLTERABDXG.sort_values(by=['CGI','E-RAB掉线率(QCI1)(小区级)'],ascending=[True,True])
df_VOLTERABDXG = df_VOLTERABDXG.drop_duplicates(['CGI'],keep='first')

df_VOLTERABDXG['vccquestiontype'] = 'VoLTE E-RAB掉线高小区(语音)-20'
df_VOLTERABDXG['vcequestiontype'] = 'VOLTERABDXG'
df_VOLTERABDXG['E-RAB掉线率(QCI1)(小区级)'] = df_VOLTERABDXG['E-RAB掉线率(QCI1)(小区级)'].apply(lambda x: format(x, '.2%'))
df_VOLTERABDXG['vcproblemtarget1'] = "[E-RAB掉线率(QCI1)(小区级)]:" + df_VOLTERABDXG['E-RAB掉线率(QCI1)(小区级)'] +\
                                     "[VoLTE话务量]:" + df_VOLTERABDXG['VoLTE语音话务量'].map(str)
df_VOLTERABDXG = df_VOLTERABDXG[['vccquestiontype','vcequestiontype','CGI','vcproblemtarget1','vcauxiliarypointer4']]

df_VOLTESXGDB = df_VOLTESXGDB.sort_values(by=['CGI','上行平均丢包率'],ascending=[True,True])
df_VOLTESXGDB = df_VOLTESXGDB.drop_duplicates(['CGI'],keep='first')
df_VOLTESXGDB['vccquestiontype'] = 'VoLTE上行高丢包小区-20'
df_VOLTESXGDB['vcequestiontype'] = 'VOLTESXGDB'
df_VOLTESXGDB['上行平均丢包率'] = df_VOLTESXGDB['上行平均丢包率'].apply(lambda x: format(x, '.2%'))
df_VOLTESXGDB['vcproblemtarget1'] = "[上行丢包率]:" + df_VOLTESXGDB['上行平均丢包率'] +"[上行丢包率采样点点数]:" +\
                                    df_VOLTESXGDB['上行总样本数'].map(str)
df_VOLTESXGDB = df_VOLTESXGDB[['vccquestiontype','vcequestiontype','CGI','vcproblemtarget1','vcauxiliarypointer4']]


df_VOLTEXXGDB = df_VOLTEXXGDB.sort_values(by=['CGI','下行平均丢包率'],ascending=[True,True])
df_VOLTEXXGDB = df_VOLTEXXGDB.drop_duplicates(['CGI'],keep='first')
df_VOLTEXXGDB['vccquestiontype'] = 'VoLTE下行高丢包小区-20'
df_VOLTEXXGDB['vcequestiontype'] = 'VOLTEXXGDB'
df_VOLTEXXGDB['下行平均丢包率'] = df_VOLTEXXGDB['下行平均丢包率'].apply(lambda x: format(x, '.2%'))
df_VOLTEXXGDB['vcproblemtarget1'] = "[下行丢包率]:" + df_VOLTEXXGDB['下行平均丢包率'] +"[下行丢包率采样点点数]:" +\
                                    df_VOLTEXXGDB['下行总样本数'].map(str)
df_VOLTEXXGDB = df_VOLTEXXGDB[['vccquestiontype','vcequestiontype','CGI','vcproblemtarget1','vcauxiliarypointer4']]

df_VOLTESXGTZ = df_VOLTESXGTZ.sort_values(by=['CGI','上行高吞字采样点占比'],ascending=[True,True])
df_VOLTESXGTZ = df_VOLTESXGTZ.drop_duplicates(['CGI'],keep='first')
df_VOLTESXGTZ['vccquestiontype'] = 'VoLTE上行高吞字小区-20'
df_VOLTESXGTZ['vcequestiontype'] = 'VOLTESXGTZ'
df_VOLTESXGTZ['上行高吞字采样点占比'] = df_VOLTESXGTZ['上行高吞字采样点占比'].apply(lambda x: format(x, '.2%'))
df_VOLTESXGTZ['vcproblemtarget1'] = "[上行高吞字率]:" + df_VOLTESXGTZ['上行高吞字采样点占比'] +"[上行丢包率采样点点数]:" +\
                                    df_VOLTESXGTZ['上行总采样点数'].map(str)
df_VOLTESXGTZ = df_VOLTESXGTZ[['vccquestiontype','vcequestiontype','CGI','vcproblemtarget1','vcauxiliarypointer4']]


df_VOLTEXXGTZ = df_VOLTEXXGTZ.sort_values(by=['CGI','下行高吞字采样点占比'],ascending=[True,True])
df_VOLTEXXGTZ = df_VOLTEXXGTZ.drop_duplicates(['CGI'],keep='first')
df_VOLTEXXGTZ['vccquestiontype'] = 'VoLTE下行高吞字小区-20'
df_VOLTEXXGTZ['vcequestiontype'] = 'VOLTEXXGTZ'
df_VOLTEXXGTZ['下行高吞字采样点占比']= df_VOLTEXXGTZ['下行高吞字采样点占比'].apply(lambda x: format(x, '.2%'))
df_VOLTEXXGTZ['vcproblemtarget1'] = "[上行高吞字率]:" + df_VOLTEXXGTZ['下行高吞字采样点占比'] +"[下行丢包率采样点点数]:" +\
                                    df_VOLTEXXGTZ['下行总采样点数'].map(str)
df_VOLTEXXGTZ = df_VOLTEXXGTZ[['vccquestiontype','vcequestiontype','CGI','vcproblemtarget1','vcauxiliarypointer4']]

df = df_VOLTEJTD.append(df_ESRVCCQHC).append(df_VOLTERABDXG).append(df_VOLTESXGDB)\
    .append(df_VOLTEXXGDB).append(df_VOLTESXGTZ).append(df_VOLTEXXGTZ)

df = df.reset_index(drop=False)

df['starttime'] = np.where(df['vcequestiontype'].isin(['VOLTEJTD','ESRVCCQHC','VOLTERABDXG']),bb,dd)
df['endtime'] = np.where((df['vcequestiontype'].isin(['VOLTEJTD','ESRVCCQHC','VOLTERABDXG'])),aa,cc)

df['vcquestioncategory'] ='小区问题点'
df['vcnetworksystem'] ='LTE'
# df['starttime'] ='2020/3/04  0:00:00'
# df['endtime'] = '2020/3/19  0:00:00'
df['flonb'] =''
df['flat'] =''
df['vcdatasource'] ='端到端信令分析优化'
df['vccgi'] = '460-00-'+df['CGI'].map(str)
df['vcroadname'] =''
df['vcfilename'] =''
df['vcspecialtag'] ='VOLTE两高两低-集团'
# df['vcproblemtarget2'] =''
df['vcproblemtarget3'] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
df['vcproblemtarget4'] =''
df['vcproblemtarget5'] ='广东'
df['vcauxiliarypointer1'] =''
df['vcauxiliarypointer2'] =''
df['vcauxiliarypointer3'] =''
df['vcauxiliarypointer5'] =''
df['vcauxiliarypointer6'] = "JT"+(df.index+1+suoyin).map(str)
df['vcproblemtarget2'] ="VOLTE-JTGD"+'-'+(time.strftime('%Y%m%d',time.localtime(time.time()))).format_map(str)+'-'+\
                        df['vcauxiliarypointer6']
df =df[['vcquestioncategory', 'vccquestiontype', 'vcequestiontype','vcnetworksystem','starttime','endtime', 'flonb',
        'flat','vcdatasource', 'vccgi', 'vcroadname', 'vcfilename', 'vcspecialtag','vcproblemtarget1',
        'vcproblemtarget2', 'vcproblemtarget3','vcproblemtarget4', 'vcproblemtarget5', 'vcauxiliarypointer1',
        'vcauxiliarypointer2', 'vcauxiliarypointer3', 'vcauxiliarypointer4', 'vcauxiliarypointer5', 'vcauxiliarypointer6']]

lieming =['vcauxiliarypointer6']
bianhao = pd.DataFrame(columns=lieming)
xuhao= df.index.max()
# print(xuhao)
bianhao.loc[0] = xuhao+1
bianhao.to_csv('D:/集团工单/bianhao.csv',header=1,encoding='gbk',index=False) #保存列名存储
df.to_csv('D:/集团工单/send.csv',header=1,encoding='gbk',index=False) #保存列名存储