from math import sin, asin, cos, radians, fabs, sqrt
EARTH_RADIUS = 6371  # 地球平均半径，6371km
def hav(theta):
    s = sin(theta / 2)
    return s * s
def get_distance_hav(lat0, lng0, lat1, lng1):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)
    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance

if __name__ == "__main__":
    print(get_distance_hav(23.276315478980543,116.42644403557391,23.276315478980543,116.4264559644261)) #计算两个坐标直线距离
    print(get_distance_hav(23.22129,113.43871,23.24416,113.47519))  # 计算两个坐标直线距离
# (23.22129,113.43871,23.24416,113.47519)


# 113.484680 23.178250