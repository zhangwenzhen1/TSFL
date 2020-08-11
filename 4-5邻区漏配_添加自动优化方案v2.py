import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import time  # 引入time模块


#@litao
# 10月17日@litao
#更新probe5.1，probe5.2两个版本的字段选择
#10月30日
#更新，增加经度、纬度的两个字段
#12月14日
#更新 在输出结果中，添加3个新的字段
#1 判断原因  (reason)     2 当前锚点小区  (now_pci_earfcn)   3  漏配NR小区频点、PCI  (omit_pci_earfcn)

#2020年2月26日 添加   nr_information，  输出 中添加    NRADD携带信息  字段



# file_path= r'C:\Users\lt\Desktop\中心经纬度.csv'
file_path= r'C:\Users\X1\Desktop\新建文件夹\移动下载DT_MS1.csv'
file_path_w= r'C:\Users\X1\Desktop\新建文件夹\45结果DT_MS1.csv'
# file_path= r'C:\Users\lt\Desktop\1120西大_空闲态拉网_MS1.csv'
# file_path_w= r'C:\Users\lt\Desktop\结果1.csv'


ticks = time.asctime( time.localtime(time.time()) )
print(ticks)
# file = pd.read_csv(file_path,low_memory=False,nrows=25000,encoding = 'gb2312',usecols=["Date & Time", "Longitude","Latitude","Event Name","Event Info",
#                                                                      "PCC Serving Cell PCI", "PCC Serving Cell RSRP(dBm)","PCC Serving Cell EARFCN",
#
#                                                                      "Detected Cell PCI", "Detected Cell RSRP(dBm)","Detected Cell EARFCN",
#                                                                     'NR Serving PCI','NR Serving DL NR-ARFCN','NR Serving SS Avg RSRP(dBm)'])

#ctrl + /

#probe5.1版本的字段
# file = pd.read_csv(file_path,low_memory=False,encoding = 'gb2312',usecols=['Date & Time', "Longitude","Latitude",'Event Name',
#                                                                      'PCC Serving Cell PCI', 'PCC Serving Cell RSRP(dBm)','PCC Serving Cell EARFCN',
#                                                                      'Event Info',
#                                                                      'Detected Cell PCI', 'Detected Cell RSRP(dBm)','Detected Cell EARFCN',
#                                                                      'NR Serving PCI','NR Serving DL NR-ARFCN','NR Serving SS Avg RSRP(dBm)'])

#probe5.2版本的字段
file = pd.read_csv(file_path,low_memory=False,encoding = 'gb2312',usecols=['Date & Time', "Longitude","Latitude",'Event Name',
                                                                     'PCC Serving Cell PCI', 'PCC Serving Cell RSRP(dBm)','PCC Serving Cell EARFCN',
                                                                     'Event Info',
                                                                     'Detected Cell PCI', 'Detected Cell RSRP(dBm)','Detected Cell EARFCN',
                                                                     'NR Serving PCI','NR Serving SSB NR-ARFCN','NR Serving SS-RSRP(dBm)'])



file.columns = ["date", "lon","la","event_name","event_info",
                "pci", "rsrp","earfcn",
                "d_pci","d_rsrp","d_earfcn",
                "nr_pci","nr_arfcn","nr_rsrp"]
print("数据读取完成")




file_event_name= file_pci_rsrp_earfcn = file_event_info = file_d_pci_rsrp_earfcn= file_nr_pci_rsrp_earfcn =pd.DataFrame(columns = ["date","lon","la", "event_name","event_info",
                "pci", "rsrp","earfcn",
                "d_pci","d_rsrp","d_earfcn",
                "nr_pci","nr_arfcn","nr_rsrp"])



file_event_name = file[ file['event_name'].notnull() ]
file_pci_rsrp_earfcn = file[(file['pci'].notnull())&(file['rsrp'].notnull())&(file['earfcn'].notnull())]
file_event_info = file[file['event_info'] .notnull()]
file_d_pci_rsrp_earfcn =  file[(file['d_pci'].notnull())&(file['d_rsrp'].notnull())&(file['d_earfcn'].notnull())]
file_nr_pci_rsrp_earfcn =file[(file['nr_pci'].notnull())&(file['nr_arfcn'].notnull())&(file['nr_rsrp'].notnull())]

print("数据复制完成取完成")



# 1秒内的信令的数量一般为15--40个，以100作为1秒的信令的最大条数
# 9月27日，在判断信令数量的基础上，添加时间判断，精确到秒
# 查找前1秒数据，即查找前100条数据，找到距离最近的符合条件的数据，若还未发现，以空记录
# 查找后4秒数据，即查找后400条数据，找到最近符合条件的数据，若未发现，记录为空
# result中保存结果
# print(type(file))
# print(file[file.event_name=='NRSCellNormalRelease'])
# print(type(file[file.event_name=='NRSCellNormalRelease']))
#
# file_event_name[file_event_name.event_name=='LTEEventB1MeasConfig']
# print("pci======",file_pci_rsrp_earfcn["pci"])
# print("earfcn====",file_pci_rsrp_earfcn["earfcn"])
# print("rsrp=======",file_pci_rsrp_earfcn["rsrp"])
# print(file[file.event_name=='LTEEventB1MeasConfig'])
# print(file[file.event_name=='LTEEventB1'])
# print(file[file.event_name=='NRSCellAddAttempt'])
# print(file[file.event_name=='LTEIntraFreqHOAttempt'])
# print(file[file.event_name=='LTERRCReestablishAttempt'])
# print(file_event_name[file_event_name.event_name=='LTEEventB1MeasConfig'])


# print("*******************************************")

def before_one_second (ind ,df,date):
    # print(df)
    ind=int(ind)
    liss = df.index.tolist()
    date =  pd.to_datetime(date)
    for num in range(ind,ind - 100 ,-1):  # 查找 当前index，及向前查找100个元素
        # print(num)
        if num in liss:
            list= df.loc[num]
            list= list.append(Series([str(num)],index=["ind"]))
            date0 = pd.to_datetime(list["date"])
            if (date - date0).seconds < 1:
                return  list
    return ( Series(['null','null','null','null',
            'null','null','null','null',
            'null','null','null','null',
            'null', 'null', 'null'],
            index=["date","lon","la", "event_name",
                "pci", "rsrp","earfcn",
                "event_info",
                "d_pci","d_rsrp","d_earfcn",
                "nr_pci","nr_arfcn","nr_rsrp","ind"]))

def after_four_second (ind ,df,date):
    ind = int(ind)
    liss = df.index.tolist()
    date =  pd.to_datetime(date)
    for num in range(ind,ind+400 ):  # 查找 当前index，及向后查找400个元素
        if num in liss:
            list = df.loc[num]
            list = list.append(Series([str(num)], index=["ind"]))
            date0 = pd.to_datetime(list["date"])
            if (date0 - date).seconds < 4:
                return  list
    return (Series(['null', 'null', 'null', 'null',
                    'null', 'null', 'null', 'null',
                    'null', 'null', 'null', 'null',
                    'null', 'null', 'null'],
                   index=["date","lon","la", "event_name",
                          "pci", "rsrp", "earfcn",
                          "event_info",
                          "d_pci", "d_rsrp", "d_earfcn",
                          "nr_pci", "nr_arfcn", "nr_rsrp", "ind"]))

def before_one_after_four_second( ind , df,date) :
    ind = int(ind)
    liss = df.index.tolist()
    date =  pd.to_datetime(date)
    for num in range(ind,ind - 100 ,-1):  # 查找 当前index，先向前查找100个元素再向后查找400个元素
        if num in liss:
            list = df.loc[num]
            list = list.append(Series([str(num)], index=["ind"]))
            date0 = pd.to_datetime(list["date"])
            if (date - date0).seconds < 1:
                return list
    for num in range(ind, ind + 400):
        if num in liss:
            list = df.loc[num]
            list = list.append(Series([str(num)], index=["ind"]))
            date0 = pd.to_datetime(list["date"])
            if (date0 - date).seconds < 4:
                return list
    return (Series(['null', 'null', 'null', 'null',
                    'null', 'null', 'null', 'null',
                    'null', 'null', 'null', 'null',
                    'null', 'null', 'null'],
                   index=["date","lon","la", "event_name",
                          "pci", "rsrp", "earfcn",
                          "event_info",
                          "d_pci", "d_rsrp", "d_earfcn",
                          "nr_pci", "nr_arfcn", "nr_rsrp", "ind"]))


result0 = pd.DataFrame(columns=['index','type',"lon","la",
                                'date_b1_config', 'b1_config_pci','b1_config_earfcn','b1_config_rsrp',
                                'date_event_b1', 'nr_pci_b1','nr_rsrp_b1',
                                'date_a3','d_pci','d_rsrp', 'd_earfcn',
                                'date_nr_attempt','nr_information','date_nr_success',
                                'pcc_pci','pcc_rsrp','pcc_earfcn',
                                'nr_pci','nr_earfcn','nr_rsrp'])



for row in  file_event_name[file_event_name.event_name=='LTEEventB1MeasConfig'].itertuples():
        # print(row)
        ind=getattr(row,'Index')
        date_b1_config=b1_config_pci=b1_config_earfcn=b1_config_rsrp=date_event_b1=nr_pci_b1=nr_rsrp_b1=date_a3=d_pci=d_rsrp=d_earfcn=date_nr_attempt=nr_information=date_nr_success=pcc_pci=pcc_rsrp=pcc_earfcn=nr_pci=nr_earfcn=nr_rsrp=''
        date= getattr(row,'date')
        event_name= getattr(row,'event_name')
        lon = getattr(row, 'lon')
        la = getattr(row, 'la')
        pci= getattr(row,'pci')
        rsrp= getattr(row,'rsrp')
        earfcn= getattr(row,'earfcn')
        event_info= getattr(row,'event_info')
        d_pci= getattr(row,'d_pci')
        d_rsrp= getattr(row,'d_rsrp')
        d_earfcn= getattr(row,'d_earfcn')
        nr_pci= getattr(row,'nr_pci')
        nr_arfcn= getattr(row,'nr_arfcn')
        nr_rsrp= getattr(row,'nr_rsrp')

        date_b1_config= date

        data_b1_config= before_one_after_four_second(ind,file_pci_rsrp_earfcn,date)
        # print(data_b1_config)
        if (data_b1_config["date"]!='null')&(data_b1_config["pci"]!='null')&(data_b1_config["rsrp"]!='null')&(data_b1_config["earfcn"]!='null'):
            b1_config_pci=data_b1_config["pci"]
            b1_config_earfcn=data_b1_config["earfcn"]
            b1_config_rsrp=data_b1_config["rsrp"]

        # print("第一字段完成")

        data_event_b1= after_four_second (ind ,file_event_name[file_event_name.event_name=='LTEEventB1'],date)
        if (data_event_b1["date"]!='null')&(data_event_b1["event_info"]!='null'):
            date_event_b1 = data_event_b1["date"]
            event_info=data_event_b1["event_info"]
            #差分，识别event info 中的nr_pci  nr_rsrp
            nr_pci_b1 = event_info
            nr_rsrp_b1 = event_info
            # nr_pci_b1=event_info.split(';')[1]
            # nr_rsrp_b1=event_info.split(';')[2]
        # print("第二字段完成")
        ind_a3 =ind
        data_a3 = after_four_second(ind, file_event_name[file_event_name.event_name == 'LTEEventA3'],date)
        # print(data_a3)
        # print(data_a3[0])
        if (data_a3["date"] != 'null')&(data_a3["ind"] != 'null'):
            date_a3 = data_a3["date"]
            ind_a3 = data_a3["ind"]
        if(ind_a3 != ind):
            data_a3_d = before_one_second (ind_a3 ,file_d_pci_rsrp_earfcn,date_a3)
            # print("第三字段完成")
            if (data_a3_d["date"] !='null')&(data_a3_d["d_pci"] !='null')&(data_a3_d["d_rsrp"] !='null')&(data_a3_d["d_earfcn"] !='null'):
                d_pci = data_a3_d["d_pci"]
                d_rsrp = data_a3_d["d_rsrp"]
                d_earfcn =data_a3_d["d_earfcn"]
        data_nr_attempt = after_four_second(ind, file_event_name[file_event_name.event_name == 'NRSCellAddAttempt'],date)
        # print("第四字段完成")
        if data_nr_attempt["date"] !='null':
            date_nr_attempt = data_nr_attempt["date"]
            nr_information = data_nr_attempt["event_info"]
        # ind_nr_success=ind
        ind_nr_success=ind
        data_nr_success = after_four_second(ind, file_event_name[file_event_name.event_name == 'NRSCellAddSuccess'],date)
        if data_nr_success["date"] != 'null':
            date_nr_success = data_nr_success["date"]
            ind_nr_success = data_nr_success["ind"]
        # print("ind=",ind,"    ind_nr_success= ",ind_nr_success)

        # if ind_nr_success != ind:

        date_nr_pcc = after_four_second(ind_nr_success, file_pci_rsrp_earfcn,date_nr_success)
        if (date_nr_pcc["date"] != 'null') & (date_nr_pcc["pci"] != 'null') & (date_nr_pcc["rsrp"] != 'null') & (
            date_nr_pcc["earfcn"] != 'null'):
            pcc_pci = date_nr_pcc["pci"]
            pcc_rsrp = date_nr_pcc["rsrp"]
            pcc_earfcn = date_nr_pcc["earfcn"]
        date_nr = after_four_second(ind_nr_success, file_nr_pci_rsrp_earfcn,date_nr_success)
        if (date_nr["date"] != 'null') & (date_nr["nr_pci"] != 'null') & (date_nr["nr_arfcn"] != 'null') & (
            date_nr["nr_rsrp"] != 'null'):
            nr_pci = date_nr["nr_pci"]
            nr_earfcn = date_nr["nr_arfcn"]
            nr_rsrp = date_nr["nr_rsrp"]



        # print("数据处理完成，正在输出")

        result0=result0.append({"index":ind, "type": "LTEEventB1MeasConfig", "lon":lon ,"la":la,
                                "date_b1_config": date_b1_config,"b1_config_pci": b1_config_pci, "b1_config_earfcn":b1_config_earfcn,"b1_config_rsrp": b1_config_rsrp ,
                                "date_event_b1":date_event_b1, "nr_pci_b1":nr_pci_b1, "nr_rsrp_b1": nr_rsrp_b1,
                                "date_a3": date_a3, "d_pci": d_pci,"d_rsrp":d_rsrp,"d_earfcn":d_earfcn,
                                "date_nr_attempt":date_nr_attempt,"nr_information":nr_information,"date_nr_success":date_nr_success,
                                "pcc_pci":pcc_pci,"pcc_rsrp":pcc_rsrp,"pcc_earfcn":pcc_earfcn,
                                "nr_pci":nr_pci,"nr_earfcn":nr_earfcn,"nr_rsrp":nr_rsrp},ignore_index=True)
        # print(result0)





# result0.columns=['序号','4-5工具切片类型','Longitude','Latitude',
#                                 '切片时间', '变更前PCI','变更前频点','变更前电平',
#                                 'EVENTB1时间', 'EVENTB1携带的NR-PCI','EVENTB1携带的NR-电平',
#                                 'EVENTA3时间','LTE最强电平小区PCI','LTE最强频点', 'LTE最强电平',
#                                 'NRADD时间','NRADDSuccess时间',
#                                 '变更后PCI','变更后电平','变更后频点',
#                                 '变更后NR -PCI','变更后NR-频点','变更后NR-电平']
# print(result0)


ticks =time.asctime( time.localtime(time.time()) )
print(ticks)
print("输出优化方案")
##1 判断原因  (reason)     2 当前锚点小区  (now_pci_earfcn)   3  漏配NR小区频点、PCI  (omit_pci_earfcn)
result0['reason']= '正常'
result0['now_pci_earfcn']= ''
result0['omit_pci_earfcn']= ''




result0_sheet_1=  result0.shift(-1)




#获取 最大的rsrp，对应的pci   主要通过max_rsrp_2_pci 实现

#max_nr_pci 取得 最大pci
def max_nr_pci (str1):
    if str1=='':
        return ''
    else:
        list = str1.split(";")
        str2 = list[2]
        list2 = str2.split(":")
        str3 = list2[1]
        list3 = str3.split(',')
        list4 = ["{:.0f}".format(float(i)+145) for i in list3]
        str4 = max(list4)
        return (str4)

#chage_2_dataframe  将str1 转化为 dataframe
def chage_2_dataframe (str1) :
    if str1=='':
        df = pd.DataFrame
        return df
    else:
        list = str1.split(";")
        str_NR_RSRP = list[2]
        str_PCI = list[1]
        # print(list)
        list_NR_RSRP = str_NR_RSRP.split(":")
        list_PCI = str_PCI.split(":")

        str_NR_RSRP_str = list_NR_RSRP[1]
        str_PCI_str = list_PCI[1]

        list_NR_RSRP_list = str_NR_RSRP_str.split(',')
        list_NR_RSRP_list = ["{:.0f}".format(float(i)+145) for i in list_NR_RSRP_list]

        list_PCI_list = str_PCI_str.split(',')
        list_PCI_list = ["{:.0f}".format(float(i)) for i in list_PCI_list]

        new_NR_RSRP = DataFrame([list_NR_RSRP_list, list_PCI_list], index=['NR_RSRP', 'NR-PCI'])
        return  new_NR_RSRP.T

def max_rsrp_2_pci(str1):
    if str1 == '':
        return  ''
    st = max_nr_pci(str1)
    df = chage_2_dataframe(str1)
    str = df[(df.NR_RSRP == st)].index.tolist()[0]
    str_value= df.loc[str, 'NR-PCI']
    return  str_value

def event_info_2_pci(str1):
    if str1 == '':
        return  ''
    else:
        str_value_list = str1.split(';')
        str1 = str_value_list[0]
        str_value_list = str1.split('PCI:')
        str_value = str_value_list[1]
    return  str_value


for row in  result0.itertuples():
    ind = getattr(row, 'Index')
    date_event_b1 = getattr(row,"date_event_b1")
    date_event_b1_sheet_1 = result0_sheet_1.loc[ind,'date_event_b1'] #  可能出现 问题 1  待检测  是否 使用当前索引来检索  检测 没问题  √
    date_nr_attempt =  getattr(row, 'date_nr_attempt')   # NRADD 时间
    nr_information =  getattr(row, 'nr_information')   # NRADD 携带的信息
    date_a3 = getattr(row, 'date_a3')   # EVENT A3  时间
    b1_config_pci = getattr(row, 'b1_config_pci')  #变更前 pci
    b1_config_earfcn= getattr(row, 'b1_config_earfcn')  # 变更前 earfcn
    nr_pci_b1 = getattr(row, 'nr_pci_b1')  # 漏NR小区频点、PCI
    nr_pci = getattr(row, 'nr_pci')  # 变更后NR -PCI
    nr_rsrp_b1 =getattr(row, 'nr_rsrp_b1')  #  EVENTB1携带的NR - 电平


    #稍后处理 遗漏最强小区
    if date_event_b1 =='':
        result0 .loc[ind,'reason'] = '无效'
    else:
        if date_event_b1 == date_event_b1_sheet_1 :
            result0.loc[ind, 'reason'] = '无效'
        else:
            if date_nr_attempt =='':
                flage_nr_attept_no = 0
                if date_a3 != '':
                    date_a3_time_format = pd.to_datetime(date_a3)
                    date_event_b1_format  = pd.to_datetime(date_event_b1)
                    if (date_a3_time_format - date_event_b1_format).seconds < 2 :
                        result0.loc[ind, 'reason'] = '碰撞A3'
                        flage_nr_attept_no =1

                if  flage_nr_attept_no == 0:
                    result0.loc[ind, 'reason'] = '漏配4-5邻区'
                    now_pci_earfcn = "PCI："+str("{:.0f}".format(b1_config_pci))+"，频点："+str("{:.0f}".format(b1_config_earfcn))
                    result0.loc[ind, 'now_pci_earfcn'] = now_pci_earfcn
                    if nr_pci_b1 !='':
                        result0.loc[ind, 'omit_pci_earfcn'] = nr_pci_b1.split(";")[1]  # 需要拆分出pci  ,已完成差分pci √
                    else:
                        result0.loc[ind, 'omit_pci_earfcn'] ='异常,EVENTB1携带的NR-PCI  列 为空 '



            else:
                max_pci =   str(max_rsrp_2_pci (nr_rsrp_b1))
                nr_pci = str("{:.0f}".format(nr_pci))
                if nr_pci =='nan':
                   nr_pci_nr_information =  event_info_2_pci(nr_information)
                   if max_pci == nr_pci_nr_information:
                       continue

                if max_pci !=nr_pci :
                    result0.loc[ind, 'reason'] = '漏配4-5最强邻区'
                    now_pci_earfcn = "PCI：" + str("{:.0f}".format(b1_config_pci)) + "，频点：" + str(
                        "{:.0f}".format(b1_config_earfcn))
                    result0.loc[ind, 'now_pci_earfcn'] = now_pci_earfcn
                    result0.loc[ind, 'omit_pci_earfcn'] = max_pci














result0.columns=['序号','4-5工具切片类型','Longitude','Latitude',
                                '切片时间', '变更前PCI','变更前频点','变更前电平',
                                'EVENTB1时间', 'EVENTB1携带的NR-PCI','EVENTB1携带的NR-电平',
                                'EVENTA3时间','LTE最强电平小区PCI','LTE最强频点', 'LTE最强电平',
                                'NRADD时间','NRADD携带信息','NRADDSuccess时间',
                                '变更后PCI','变更后电平','变更后频点',
                                '变更后NR -PCI','变更后NR-频点','变更后NR-电平',
                                '判断原因','当前锚点小区','漏配NR小区频点、PCI']

result0.to_csv(file_path_w, sep=',',encoding = 'gb2312', header=True, index=True)
ticks =time.asctime( time.localtime(time.time()) )
print(ticks)