import math
# print("math.log(100.12) : ", math.log(100.12))
# b= 153.8
# b= 1
# c= 2.08
# x= 288

# t = (-0.001/math.log(10,math.e))*(x*math.log(x,math.e)-x) +b*math.log(x,math.e)+c*x
# print(t)

# # x2 =0.00000000000000000000000000000000000000000000000000000000000000000000000000000001
# x2 =0.0031
# t2 = (-1/math.log(10,math.e))*(x2*math.log(x2,math.e)-x2) +b*math.log(x2,math.e)+c*x2
# print("总面积:",t-t2)
# print("09.总面积:",0.99*(t-t2))
#
# # x3 = 12.388
# x3 = 12
# t3 = (-1/math.log(10,math.e))*(x3*math.log(x3,math.e)-x3) +b*math.log(x3,math.e)+c*x3
# print("测试",t3)
# print("计算09.总面积:",0.99*(t3-t2))
# print("计算0.99面积概率",(t3-t2)/(t-t2))
a= -4.232
b= 1.909
c= 980.9
# x= 0.5
# y = a*math.log(x)+c/x + b
#
# print(y)
# a = -1.0578139391495305
# b = 6.031
# c = 900000000.41947178614692
def func(x):
    t = (a/math.log(10,math.e))*(x*math.log(x,math.e)-x) +c*math.log(x,math.e)+b*x
    return t
y  = func(0.03)
y1 = func(288)
y2 = func(14.1)
y3 = func(43.75)
print("x=1: t=",y)
print("24小时: t1=",y1)
print("1小时: t2=",y2)
print("x[1]/x[24]:",(y2-y)/(y1-y))
print("4小时",y3)
print("x[4]/x[24]:",(y3-y)/(y1-y))