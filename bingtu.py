from pyecharts import Line, Bar, Pie, EffectScatter
# import pandas as pd
#
# df = pd.read_csv('D:\Volte\S1异常表整理.csv', encoding='gbk')
# df = df[['temp2.msisdn','temp2.procedure_status']]
# print(df.iloc[:,0].size)
# df0 = df.loc[( df['temp2.procedure_status'] == 0)]
# df0 = df0.drop_duplicates(['temp2.msisdn'])
# print(df0.iloc[:,0].size)
#
# df1 = df.loc[( df['temp2.procedure_status'] == 1)]
# df1 = df1.drop_duplicates(['temp2.msisdn'])
# print(df1.iloc[:,0].size)
#
# df255 = df.loc[( df['temp2.procedure_status'] == 255)]
# df255 = df255.drop_duplicates(['temp2.msisdn'])
# print(df255.iloc[:,0].size)
#
# df0 = df0.loc[~df0['temp2.msisdn'].isin(df1['temp2.msisdn'])]
# print(df0.iloc[:,0].size)
#
# df0 = df0.loc[~df0['temp2.msisdn'].isin(df255['temp2.msisdn'])]
#
# print(df0.iloc[:,0].size)
# import matplotlib.pyplot as plt
# attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
# v1 = [11, 12, 13, 10, 10, 10]
# pie = Pie("饼图示例",title_pos='center',width=900)
# pie.add(
#     "",
#     attr,
#     v1,
#     # center=[75, 50],
#     is_label_show=True,
#     is_more_utils=True,
#     label_text_color=None,
#     # legend_orient="vertical",
#     # legend_pos="left",
# )
#
# pie.render(path="Bing1.html")



# attr = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
# v1 = [11, 12, 13, 10, 10, 10]
# pie = Pie("饼图-圆环图示例", title_pos='center')
# pie.add(
#     "",
#     attr,
#     v1,
#     radius=[40, 75],
#     label_text_color=None,
#     is_label_show=True,
#     is_more_utils=True,
#     legend_orient="vertical",
#     legend_pos="left",
# )
# pie.render(path="Bing2.html")


# import numpy as np
# import matplotlib.pyplot as plt
# labels = 'A', 'B', 'C', 'D'
# fracs = [15, 30.55, 44.44, 10]
# explode = [0, 0, 0, 0] # 0.1 凸出这部分，
# plt.axes(aspect=0) # set this , Figure is round, otherwise it is an ellipse
# # autopct ，show percet
# plt.pie(x=fracs, labels=labels, explode=explode, autopct='%3.1f %%',
#     shadow=True, labeldistance=1.1, startangle=90, pctdistance=0.6 )
# '''
# labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
# autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
# shadow，饼是否有阴影
# startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
# pctdistance，百分比的text离圆心的距离
# patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本
# '''
# plt.show()
# 数据
attr =["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
v1 =[5, 20, 36, 10, 10, 100]
v2 =[55, 60, 16, 20, 15, 80]

bar = Bar('柱形图', '库存量')
bar.add('服装', attr, v1,  is_label_show=True)
bar.show_config()
bar.render(path='D:/PycharmProjects/python/01-01柱形图.html')

bar2 = Bar("显示标记线和标记点")
bar2.add('商家A', attr, v1, mark_point=['avgrage'])
bar2.add('商家B', attr, v2, mark_point=['min', 'max'])
bar2.show_config()
bar2.render(path='D:\PycharmProjects\python/01-02标记点柱形图.html')

bar3 = Bar("水平显示")
bar3.add('商家A', attr, v1)
bar3.add('商家B', attr, v2, is_convert=True)
bar3.show_config()
bar3.render(path='D:\PycharmProjects\python/01-03水平柱形图.html')

# 普通折线图
line = Line('折线图')
line.add('商家A', attr, v1, mark_point=['max'])
line.add('商家B', attr, v2, mark_point=['min'], is_smooth=True)
line.show_config()
line.render(path='D:\PycharmProjects\python/01-04折线图.html')

# 阶梯折线图
line2 = Line('阶梯折线图')
line2.add('商家A', attr, v1,  is_step=True, is_label_show=True)
line2.show_config()
line2.render(path='D:\PycharmProjects\python/01-05阶梯折线图.html')

# 面积折线图
line3 =Line("面积折线图")
line3.add("商家A", attr, v1, is_fill=True, line_opacity=0.2,   area_opacity=0.4, symbol=None, mark_point=['max'])
line3.add("商家B", attr, v2, is_fill=True, area_color='#a3aed5', area_opacity=0.3, is_smooth=True)
line3.show_config()
line3.render(path='D:\PycharmProjects\python/01-06面积折线图.html')