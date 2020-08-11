import math
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist #导入坐标轴加工模块
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

fig=plt.figure(figsize=(6,4)) #新建画布
ax=axisartist.Subplot(fig,111) #使用axisartist.Subplot方法创建一个绘图区对象ax
fig.add_axes(ax) #将绘图区对象添加到画布中
dd =2.000023
qq =1
def logarithm_func(x, a=math.e): #定义指数函数
    y = dd*x*math.log(x, a) +x
    return y

X = np.linspace(0.01, 4, 100) #构造自变量组
Y=[logarithm_func(x) for x in X] #求函数值
ax.plot(X, Y, label=r'$0<a<1$') #绘制指数函数
ax.scatter(1, 0, color='red')
plt.legend()
plt.show()

print(max(X), max(Y)) #测试一下自变量最大值和因变量最大值，为后面的坐标轴设置依据
ax.axis[:].set_visible(False) #隐藏原来的实线矩形
ax.axis["x"]=ax.new_floating_axis(0, 0, axis_direction="bottom") #添加x轴
ax.axis["y"]=ax.new_floating_axis(1, 0, axis_direction="bottom") #添加y轴

ax.axis["x"].set_axisline_style("-|>", size=1.0) #给x坐标轴加箭头
ax.axis["y"].set_axisline_style("-|>", size=1.0) #给y坐标轴加箭头

ax.annotate(s='x', xy=(max(X), 0), xytext=(max(X)+1, 0.3)) #标注x轴
ax.annotate(s='y', xy=(0, 1.0), xytext=(-0.2, max(Y)+1)) #标注y轴

plt.xlim(0, 5) #设置横坐标范围
plt.ylim(0, 3) #设置纵坐标范围
X_lim=np.arange(int(min(X)), max(X)+1, 1)
ax.set_xticks(X_lim) #设置x轴刻度
Y_lim=np.arange(-3, max(Y)+1, 1)
ax.set_yticks(Y_lim) #设置y轴刻度
ax.annotate(s=r'$y=log_a x$',xy=(1, 1), xytext=(1, 1)) #r'$y=(\frac{1}{2})^x$'
