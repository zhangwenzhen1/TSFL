
import pandas as pd
from math import sin, asin, cos, radians, fabs, sqrt
import numpy as np

EARTH_RADIUS = 6371  # 地球平均半径，6371km


def hav(theta):
    s = np.sin(theta / 2)
    return s * s


def get_distance_hav(lat0, lng0, lat1, lng1):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = np.radians(lat0)
    lat1 = np.radians(lat1)
    lng0 = np.radians(lng0)
    lng1 = np.radians(lng1)
    dlng = np.fabs(lng0 - lng1)
    dlat = np.fabs(lat0 - lat1)
    h = hav(dlat) + np.cos(lat0) * np.cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * (np.arcsin(np.sqrt(h)))
    return distance


# eptable = pd.read_csv('D:\集团工单/v_eptable.csv', encoding='utf-8', usecols=['cgi', 'cellname', 'longitude', 'latitude'])
#
# df = pd.read_csv('D:\集团工单/vTD.csv', encoding='gbk')
# df['真实投诉异常小区CGI'] = df['真实投诉异常小区CGI'].astype(np.str)
# df['真实投诉异常小区CGI'] = df['真实投诉异常小区CGI'].apply(lambda x: x.replace('460-00-', ''))
# result = pd.merge(df, eptable, on='cgi', how='left', suffixes=('', '_y'))  # pandas csv表左连接
# result.rename(columns={'cellname': '平台发现小区中文名'}, inplace=True)
#
# result = pd.merge(result, eptable, left_on='真实投诉异常小区CGI', right_on='cgi', how='left',
#                   suffixes=('', '_y'))  # pandas csv表左连接
# result = result.loc[
#     (result['longitude'].notnull()) & (result['latitude'].notnull()) & (result['longitude_y'].notnull()) & (
#         result['latitude_y'].notnull())]
# result['站间距'] = get_distance_hav(result['latitude'], result['longitude'], result['latitude_y'], result['longitude_y'])
# result.drop(['cgi_y'], axis=1, inplace=True)
# result = result[['cgi', '平台发现小区中文名', 'longitude', 'latitude', '真实投诉异常小区中文名称', '真实投诉异常小区CGI',
#                  'longitude_y', 'latitude_y', '站间距']]
# result.to_csv('D:\集团工单/result.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
# #
# print(result.columns)
# # print(eptable.head())
# print(result.head())
# print(230829129//256)
# print(230829129%256)
# print(177109771//256)
# print(177109771%256)
#
# print(8773953%256)
# print(8773953//256)
# path = 'D:\集团工单/0_327条工单再分析.xlsx'
# df = pd.read_excel(path, sheet_name='重整站间距',usecols=['平台CGI', '地市落实CGI', '平台-1小区(S1)补充诺西的', '平台-2小区',
#                                                      '平台-3小区','平台-4小区','平台-5小区'])

# def choose(date,a,b,c,d):
#     date = date.loc[(date[a].notnull()) & (date[b].notnull()) & (date[c].notnull()) & (date[d].notnull())]
#     return date
#
# eptable = pd.read_csv('D:\集团工单/v_eptable.csv', encoding='utf-8', usecols=['cgi', 'cellname', 'longitude', 'latitude'])
# eptable = eptable.drop_duplicates('cgi')
# df = pd.read_csv('D:\集团工单/512-518期间广州的故障小区12.csv', encoding='gbk')
# print(df.columns)
# print(df.iloc[:, 0].size)
# date = choose(df,'longitude','latitude','longitude_live','latitude_live')
# print(date.iloc[:, 0].size)
# date['落实小区与常驻小区站间距_live'] = get_distance_hav(date['latitude'], date['longitude'], date['latitude_live'], date['longitude_live'])
#
# date['落实小区与常驻小区站间距_work']= get_distance_hav(date['latitude'], date['longitude'], date['latitude_work'], date['longitude_work'])
# print(date.iloc[:, 0].size)
# date.to_csv('D:\集团工单/512-518期间广州的故障小区123.csv', header=1, encoding='gbk', index=True)  # 保存列名存储
# print(df.iloc[:, 0].size)
# df = df.drop_duplicates()
# print(df.iloc[:, 0].size)
# print(df.columns)
# df_1 = pd.read_csv('D:\集团工单/temp2.csv', encoding='gbk')
# print(df_1.iloc[:, 0].size)
# df_1 = df_1.drop_duplicates()
# print(df_1.iloc[:, 0].size)
# print(df_1.head())
# # df = pd.read_excel(path, sheet_name='重整站间距',)
# df_1['常驻小区'] = df_1['常驻小区'].astype(np.str)
# df_1['常驻小区'] = df_1['常驻小区'].apply(lambda x: x.replace('460-00-', ''))
# result = pd.merge(df, df_1, on='受理工单号',how='left', suffixes=('', '_luoshi'))  # pandas csv表左连接
# print(result.iloc[:, 0].size)
# print(result.columns)
# result = pd.merge(result, eptable, left_on='地市落实CGI',right_on='cgi', how='left', suffixes=('', '_落实'))  # pandas csv表左连接
# print(result.iloc[:, 0].size)
# print(result.columns)
# result = pd.merge(result, eptable, left_on='常驻小区',right_on='cgi', how='left', suffixes=('', '_常驻'))  # pandas csv表左连接
# print(result.iloc[:, 0].size)
# print(result.columns)

# date = choose(result,'longitude','latitude','longitude_常驻','latitude_常驻')
# date['落实小区与常驻小区站间距'] = get_distance_hav(date['latitude'], date['longitude'], date['latitude_常驻'], date['longitude_常驻'])
# result = pd.merge(result, eptable, left_on='平台-1小区(S1)补充诺西的',right_on='cgi', how='left', suffixes=('', '_平台1'))  # pandas csv表左连接
# result = pd.merge(result, eptable, left_on='平台-2小区',right_on='cgi', how='left', suffixes=('', '_平台2'))  # pandas csv表左连接
# result = pd.merge(result, eptable, left_on='平台-3小区',right_on='cgi', how='left', suffixes=('', '_平台3'))  # pandas csv表左连接
# result = pd.merge(result, eptable, left_on='平台-4小区',right_on='cgi', how='left', suffixes=('', '_平台4'))  # pandas csv表左连接
# result = pd.merge(result, eptable, left_on='平台-5小区',right_on='cgi', how='left', suffixes=('', '_平台5'))  # pandas csv表左连接
# print(df.iloc[:, 0].size)
# print(result.iloc[:, 0].size)
# result.drop(['cgi'], axis=1, inplace=True)
# print(result.columns)

# date = choose(result,'longitude','latitude','longitude_平台','latitude_平台')
# date1 = choose(result,'longitude','latitude','longitude_平台1','latitude_平台1')
# date2 = choose(result,'longitude','latitude','longitude_平台2','latitude_平台2')
# date3 = choose(result,'longitude','latitude','longitude_平台3','latitude_平台3')
# date4 = choose(result,'longitude','latitude','longitude_平台4','latitude_平台4')
# date5 = choose(result,'longitude','latitude','longitude_平台5','latitude_平台5')
# # date.to_csv('D:\集团工单/date.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
# print(date.info())
# # date['latitude'] = date['latitude'].astype(np.float64)
# # date['longitude'] = date['longitude'].astype(np.float64)
# # date['latitude_平台'] = date['latitude_平台'].astype(np.float64)
# # date['latitude_平台'] = date['latitude_平台'].astype(np.float64)
# date['平台小区站间距'] = get_distance_hav(date['latitude'], date['longitude'], date['latitude_平台'], date['longitude_平台'])
# date = date[['平台小区站间距']]
#
#
# date1['平台1小区站间距'] = get_distance_hav(date1['latitude'], date1['longitude'], date1['latitude_平台1'], date1['longitude_平台1'])
# date1 = date1[['平台1小区站间距']]
# dff = pd.merge(result,date1,right_index=True, left_index=True, how ='outer', suffixes=('', '_y'))
#
# date2['平台2小区站间距'] = get_distance_hav(date2['latitude'], date2['longitude'], date2['latitude_平台2'], date2['longitude_平台2'])
# date2 = date2[['平台2小区站间距']]
#
# date3['平台3小区站间距'] = get_distance_hav(date3['latitude'], date3['longitude'], date3['latitude_平台3'], date3['longitude_平台3'])
# date3 = date3[['平台3小区站间距']]
#
# date4['平台4小区站间距'] = get_distance_hav(date4['latitude'], date4['longitude'], date4['latitude_平台4'], date4['longitude_平台4'])
# date4 = date4[['平台4小区站间距']]
#
# date5['平台5小区站间距'] = get_distance_hav(date5['latitude'], date5['longitude'], date5['latitude_平台5'], date5['longitude_平台5'])
# date5 = date5[['平台5小区站间距']]
# # print(date.head())
# dff = pd.merge(result,date,right_index=True, left_index=True, how ='outer', suffixes=('', '_y'))
# dff = pd.merge(dff,date1,right_index=True, left_index=True, how ='outer', suffixes=('', '_y'))
# dff = pd.merge(dff,date2,right_index=True, left_index=True, how ='outer', suffixes=('', '_y'))
# dff = pd.merge(dff,date3,right_index=True, left_index=True, how ='outer', suffixes=('', '_y'))
# dff = pd.merge(dff,date4,right_index=True, left_index=True, how ='outer', suffixes=('', '_y'))
# dff = pd.merge(dff,date5,right_index=True, left_index=True, how ='outer', suffixes=('', '_y'))
# date.to_csv('D:\集团工单/落实小区与常驻小区站间距.csv', header=1, encoding='gbk', index=True)  # 保存列名存储
def jisuan(df,c,*a):
    df = df.groupby(*a, as_index=True)[c].agg(np.sum)
    b = df.reset_index(drop=False)
    return b
df = pd.read_csv('D:\集团工单/back.csv', encoding='gbk')
print(df.columns)
c='voltetraval','volteradconnratio'
b = jisuan(df,c,['city','startdate'])
print(b.head())
