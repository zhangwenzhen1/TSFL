import numpy as np
import math
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

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


def returnSquarePoint(lon, lat, dis=1500):
    '''
    :param lon:经度
    :param lat:纬度
    :param dis:需要的正方形边长
    :return:4个角经纬度的列表  [(),(),(),()]
    '''
    L = []
    dis = dis / 2
    perimeter_lon = 2 * 6378137 * math.pi
    north_south_lon_m = 360 / perimeter_lon  # 南北方向一米,单位（度）

    perimeter_lat = 40075016.68557849 * math.cos(lon)
    east_west_lat_m = 360 / perimeter_lat  # 东西方向一米,单位（度on
    maxlat = lat + north_south_lon_m * dis
    minlon = lon - east_west_lat_m * dis
    maxlon = lon + east_west_lat_m * dis
    minlat = lat - north_south_lon_m * dis
    L.append(maxlat)
    L.append(minlat)
    L.append(maxlon)
    L.append(minlon)
    return L


def choosedate(df, L):
    df = df.loc[
        ((df['longitude'] > L[3]) & (df['longitude'] < L[2])) & ((df['latitude'] > L[1]) & (df['latitude'] < L[0]))]
    return df


def save(Result, j):
    Result.rename(columns={'主小区cgi': 'cgi', 'cgi': 'ncgi', '小区间距离': 'distance'}, inplace=True)
    Result = Result[['cgi', 'ncgi', 'distance']]
    filename = 'julitable' + str(j)
    Result.to_csv('D:\集团工单\eptable' + '/' + filename + '.csv', header=1, encoding='gbk', index=False)
    lieming = ['cgi', 'ncgi', '小区间距离']
    Result = pd.DataFrame(columns=lieming)
    return Result


df = pd.read_csv('D:\集团工单\eptable\eptable.csv', encoding='utf-8', usecols=['cityname', 'cgi', 'longitude', 'latitude'])
df['flag'] = 0
print(df.iloc[:, 0].size)
df = df.loc[(df['cityname'] == "佛山")]
print(df.iloc[:, 0].size)
df = df.drop_duplicates('cgi')
# 构造输出结果表
lieming = ['cgi', 'ncgi', '小区间距离']
Result = pd.DataFrame(columns=lieming)
i = 0
j = 1
while i < df.iloc[:, 0].size:

    a = df.iloc[i, :]
    # df['flag'] = np.where(df['cgi'] == a[1], 1, df['flag'])
    # print(a[0],a[1],a[2],a[3])
    L = returnSquarePoint(a[2], a[3], dis=1500)
    df1 = choosedate(df, L)
    df1 = df1.loc[df1['cgi'] != a['cgi']]
    df1['主小区cgi'] = a[1]
    df1['lon'] = a[2]
    df1['lat'] = a[3]
    df1['小区间距离'] = get_distance_hav(df1['lat'], df1['lon'], df1['latitude'], df1['longitude'])
    df1 = df1.loc[df1['小区间距离'] < 1]
    df1['falg'] = i
    df1 = df1[['主小区cgi', 'cgi', '小区间距离']]
    i += 1
    # df = df.loc[df['flag'] != 1]
    # df = df.loc[(~df['cgi'].isin(df1['主小区cgi']))]
    # df.reset_index(drop=True, inplace=True)
    Result = Result.append(df1)
    j += 1
    if i % 5000 == 0 and i > 0:
        print(i)
        # if Result.iloc[:, 0].size % 50000 ==0 and Result.iloc[:, 0].size > 0:
        Result = save(Result, j)
    # print(df1.head(100))
# Result = Result[['cityname','主小区cgi', 'lon', 'lat', 'cgi', 'longitude', 'latitude', '小区间距离', 'falg']]
# Result.rename(columns={'主小区cgi': 'cgi','cgi': 'ncgi','小区间距离': 'distance'}, inplace=True)
# Result = Result[['cgi','ncgi','distance']]
# # 将结果存在csv中
# Result.to_csv('D:\集团工单\eptable\julitable2.csv', header=1, encoding='gbk', index=False)

# #输入文件路径
# input_file = "D:\集团工单\eptable\eptable - 副本.csv"
# #输出文件路径
# output_file = "D:\Test\output.csv"
# # f = open(input_file, 'r',encoding='utf-8')
# while True:
#     con = f.readline()
#     print()
#     if len(con) == 0:
#         break
