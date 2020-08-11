import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.optimize import leastsq
# from sklearn.linear_model import LinearRegression
from scipy import sparse
import numpy as np
from scipy import log
from scipy.optimize import curve_fit
import math
from matplotlib.pyplot import MultipleLocator
font = FontProperties()
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
df = pd.read_csv('D:\Volte\zhixindu.csv', encoding='gbk')

# #绘图
# plot1 = plt.plot(df['interval'], df['request_num'], 's',label='original values')
# plt.xlabel('interval')
# plt.ylabel('request_num')
# #把x轴的刻度间隔设置为1，并存在变量里
# x_major_locator=MultipleLocator(1)
#
# #把y轴的刻度间隔设置为10，并存在变量里
# y_major_locator=MultipleLocator(10)
#
# ax=plt.gca()
# #ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)
# #把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)
# #把y轴的主刻度设置为10的倍数
#
# plt.xlim(0,300)
# # plt.legend(loc=4) #指定legend的位置右下角
# plt.title('异常话单数匹配关系')
# plt.show()

y = df['request_num']
y= np.array(y)
x = df['interval']
x= np.array(x)
# print(type(x))
# print(x)

# 拟合函数
def func(x,a,b,c):
      return a*np.log(x)+c/(x) + b
     # return a**x+c/(1+x) + b
     # return c/(1+a*math.e**b*x)
popt, pcov = curve_fit(func, x, y)

a = popt[0]#popt里面是拟合系数，读者可以自己help其用法
b = popt[1]
c = popt[2]
print(a,b,c)
yvals = func(x,a,b,c)
### Calculate R Square ###
calc_ydata = [func(i, popt[0], popt[1],popt[2]) for i in x]
res_ydata = np.array(y) - np.array(calc_ydata)
ss_res = np.sum(res_ydata ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

plot1=plt.plot(x, y, '*',label='original values')
plot2=plt.plot(x, yvals, 'r',label='curve_fit values')
plt.xlabel('时间 t/5min')
plt.ylabel('VoLTE语音投诉工单数/个')
plt.legend(loc='upper right')#指定legend的位置,读者可以自己help它的用法
plt.xlim(0,300)
plt.ylim(0,135)
# #把x轴的刻度间隔设置为1，并存在变量里
x_major_locator=MultipleLocator(5)
#把y轴的刻度间隔设置为10，并存在变量里
y_major_locator=MultipleLocator(5)
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator) #把x轴的主刻度设置为1的倍数

ax.yaxis.set_major_locator(y_major_locator) #把y轴的主刻度设置为10的倍数

plt.title('curve_fit')
plt.show()
plt.savefig('p2.png')

#使用非线性最小二乘法拟合
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# import numpy as np
# #用指数形式来拟合
# x = np.arange(1, 17, 1)
# y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
#
# print(type(x))
# print(x)
# def func(x,a,b):
#     return a*np.exp(b/x)
# popt, pcov = curve_fit(func, x, y)
# a=popt[0]#popt里面是拟合系数，读者可以自己help其用法
# b=popt[1]
# yvals=func(x,a,b)
# plot1=plt.plot(x, y, '*',label='original values')
# plot2=plt.plot(x, yvals, 'r',label='curve_fit values')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
# plt.title('curve_fit')
# plt.show()
# plt.savefig('p2.png')


# 用训练集进行拟合优度，验证回归方程是否合理
### Calculate R Square ###
