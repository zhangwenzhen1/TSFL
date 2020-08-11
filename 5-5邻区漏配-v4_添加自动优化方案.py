import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import time  # 引入time模块
from math import isnan

# 10月17日@litao
#更新probe5.1，probe5.2两个版本的字段选择
#10月27日
#更新，增加经度、纬度的两个字段
#变更前NR电平与变更前NR PCI 字段输出互换，LTE最强频点与LTE最强电平 字段互换
#NR最强小区频点   event_info 中没有earfcn值，在前一秒内查找“NR Detected SSB NR-ARFCN  ”

# file_path= r'C:\Users\lt\Desktop\中心经纬度.csv'
file_path= r'C:\Users\X1\Desktop\新建文件夹\移动下载DT3_MS1.csv'
# file_path= r'C:\Users\lt\Desktop\转换后LOG\样例数据.csv'
file_path_w= r'C:\Users\X1\Desktop\新建文件夹\结果DT3_MS1.csv'
ticks = time.asctime( time.localtime(time.time()) )
print(ticks)
#
# file = pd.read_csv(file_path,low_memory=False,nrows=5000,encoding = 'gb2312',usecols=["Date & Time", "Longitude","Latitude", "Event Name","Event Info",
#                                                                     "PCC Serving Cell PCI", "PCC Serving Cell RSRP(dBm)","PCC Serving Cell EARFCN",
#                                                                      "Detected Cell PCI", "Detected Cell RSRP(dBm)","Detected Cell EARFCN",
#                                                                      "NR Serving PCI","NR Serving SSB NR-ARFCN","NR Serving SS-RSRP(dBm)",
#                                                                      "NR Detected PCI","NR Detected SSB NR-ARFCN","NR Detected SS-RSRP(dBm)"])


#ctrl + /
#probe5.1版本的字段
# file = pd.read_csv(file_path,low_memory=False,encoding = 'gb2312',usecols=["Date & Time", "Longitude","Latitude", "Event Name","Event Info",
#                                                                      "PCC Serving Cell PCI", "PCC Serving Cell RSRP(dBm)","PCC Serving Cell EARFCN",
#                                                                      "Detected Cell PCI", "Detected Cell RSRP(dBm)","Detected Cell EARFCN",
#                                                                      "NR Serving PCI","NR Serving DL NR-ARFCN","NR Serving SS Avg RSRP(dBm)",
#                                                                      "NR Detected PCI","NR Detected DL NR-ARFCN","NR Detected SS Avg RSRP(dBm)"])


#probe5.2版本的字段

file = pd.read_csv(file_path,low_memory=False,encoding = 'gb2312',usecols=["Date & Time", "Longitude","Latitude","Event Name","Event Info",
                                                                     "PCC Serving Cell PCI", "PCC Serving Cell RSRP(dBm)","PCC Serving Cell EARFCN",
                                                                     "Detected Cell PCI", "Detected Cell RSRP(dBm)","Detected Cell EARFCN",
                                                                     "NR Serving PCI","NR Serving SSB NR-ARFCN","NR Serving SS-RSRP(dBm)",
                                                                     "NR Detected PCI","NR Detected SSB NR-ARFCN","NR Detected SS-RSRP(dBm)"])


file.columns = ["date","lon","la", "event_name","event_info",
                "pci", "rsrp","earfcn",
                "d_pci","d_rsrp","d_earfcn",
                "nr_pci","nr_arfcn","nr_rsrp",
                "nr_d_pci","nr_d_arfcn","nr_d_rsrp"]


print("数据读取完成")



file_event_name = file[ file['event_name'].notnull() ]#保存event_name不为空的数据
file_pci_rsrp_earfcn = file[(file['pci'].notnull())&(file['rsrp'].notnull())&(file['earfcn'].notnull())] #保存pci,rsrp,earfcn不为空的数据
file_event_info = file[file['event_info'] .notnull()] #保存event_info不为空的数据
file_d_pci_rsrp_earfcn =  file[(file['d_pci'].notnull())&(file['d_rsrp'].notnull())&(file['d_earfcn'].notnull())] #保存d_pci,d_rsrp,d_earfcn不为空的数据
file_nr_pci_rsrp_earfcn =file[(file['nr_pci'].notnull())&(file['nr_arfcn'].notnull())&(file['nr_rsrp'].notnull())] #保存nr_pci,nr_rsrp,nr_earfcn不为空的数据
file_nr_d_pci_rsrp_earfcn =file[(file['nr_d_pci'].notnull())&(file['nr_d_arfcn'].notnull())&(file['nr_d_rsrp'].notnull())] #保存nr_d_pci,nr_d_rsrp,nr_d_earfcn不为空的数据
# print(file_nr_d_pci_rsrp_earfcn)




# print(file)

# 1秒内的信令的数量一般为15--40个，以100作为1秒的信令的最大条数
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
            list= list.append(Series([str(ind)],index=["ind"]))
            date0 = pd.to_datetime(list["date"])
            if (date - date0).seconds < 1:
                return  list
    return ( Series(['null','null','null','null','null','null',
            'null','null','null','null',
            'null','null','null','null',
            'null', 'null', 'null',
            'null'],
            index=["date","lon","la", "event_name",
                "pci", "rsrp","earfcn",
                "event_info",
                "d_pci","d_rsrp","d_earfcn",
                "nr_pci","nr_arfcn","nr_rsrp",
                "nr_d_pci", "nr_d_arfcn", "nr_d_rsrp",
                "ind"]))

def after_four_second (ind ,df,date):
    ind = int(ind)
    liss = df.index.tolist()
    date = pd.to_datetime(date)
    for num in range(ind,ind+400 ):  # 查找 当前index，及向后查找400个元素
        if num in liss:
            list = df.loc[num]
            list = list.append(Series([str(ind)], index=["ind"]))
            date0 = pd.to_datetime(list["date"])
            if (date0 - date).seconds < 4:
                return list
    return ( Series(['null','null','null','null','null','null',
            'null','null','null','null',
            'null','null','null','null',
            'null', 'null', 'null',
            'null'],
            index=["date","lon","la", "event_name",
                "pci", "rsrp","earfcn",
                "event_info",
                "d_pci","d_rsrp","d_earfcn",
                "nr_pci","nr_arfcn","nr_rsrp",
                "nr_d_pci", "nr_d_arfcn", "nr_d_rsrp",
                "ind"]))


def before_one_after_four_second( ind , df,date) :
    ind = int(ind)
    liss = df.index.tolist()
    date = pd.to_datetime(date)
    for num in range(ind,ind - 100 ,-1):  # 查找 当前index，先向前查找100个元素再向后查找400个元素
        if num in liss:
            list = df.loc[num]
            list = list.append(Series([str(ind)], index=["ind"]))
            date0 = pd.to_datetime(list["date"])
            if (date - date0).seconds < 1:
                return list
    for num in range(ind, ind + 400):
        if num in liss:
            list = df.loc[num]
            list = list.append(Series([str(ind)], index=["ind"]))
            date0 = pd.to_datetime(list["date"])
            if (date0 - date).seconds < 4:
                return list
    return ( Series(['null','null','null','null','null','null',
            'null','null','null','null',
            'null','null','null','null',
            'null', 'null', 'null',
            'null'],
            index=["date","lon","la", "event_name",
                "pci", "rsrp","earfcn",
                "event_info",
                "d_pci","d_rsrp","d_earfcn",
                "nr_pci","nr_arfcn","nr_rsrp",
                "nr_d_pci", "nr_d_arfcn", "nr_d_rsrp",
                "ind"]))



result0 = pd.DataFrame(columns=['index','type',"lon","la",
                                'date_nr_a3',
                                'nr_a3_pci','nr_a3_earfcn','nr_a3_rsrp',
                                'nr_a3_nr_rsrp','nr_a3_nr_pci','nr_a3_nr_arfcn',
                                'date_nr_a3_copy',
                                'nr_a3_d_pci','nr_a3_d_rsrp', 'nr_a3_d_earfcn',
                                'date_a3',
                                'd_pci','d_earfcn','d_rsrp',
                                'date_nrs_attempt','date_nrs_success',
                                'pcc_pci','pcc_rsrp','pcc_earfcn',
                                'nr_pci','nr_earfcn','nr_rsrp'
                                ])

for row in  file_event_name[file_event_name.event_name=='NREventA3'].itertuples():
        # print(row)
        ind=getattr(row,'Index')
        date_nr_a3=nr_a3_pci=nr_a3_earfcn=nr_a3_rsrp=  nr_a3_nr_pci= nr_a3_nr_arfcn= nr_a3_nr_rsrp= nr_a3_d_pci=nr_a3_d_rsrp=nr_a3_d_earfcn= date_a3=d_pci=d_rsrp=d_earfcn=date_nrs_attempt=date_nrs_success =pcc_pci=pcc_rsrp=pcc_earfcn=nr_pci=nr_earfcn=nr_rsrp=''
        date= getattr(row,'date')
        lon = getattr(row,'lon')
        la =getattr(row,'la')
        event_name= getattr(row,'event_name')
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
        nr_d_pci = getattr(row, 'nr_d_pci')
        nr_d_arfcn = getattr(row, 'nr_d_arfcn')
        nr_d_rsrp = getattr(row, 'nr_d_rsrp')


        date_nr_a3= date

        data_nr_a3= before_one_after_four_second(ind,file_pci_rsrp_earfcn,date)
        # print(date_nr_a3)
        if (data_nr_a3["date"]!='null')&(data_nr_a3["pci"]!='null')&(data_nr_a3["rsrp"]!='null')&(data_nr_a3["earfcn"]!='null'):
            nr_a3_pci=data_nr_a3["pci"]
            nr_a3_earfcn=data_nr_a3["earfcn"]
            nr_a3_rsrp=data_nr_a3["rsrp"]
        data_nr_a3_d = before_one_second(ind, file_nr_pci_rsrp_earfcn,date)
        if (data_nr_a3_d["date"] != 'null') & (data_nr_a3_d["nr_pci"] != 'null') & (data_nr_a3_d["nr_rsrp"] != 'null') & (
                data_nr_a3_d["nr_arfcn"] != 'null'):
            nr_a3_nr_pci = data_nr_a3_d["nr_pci"]
            nr_a3_nr_rsrp = data_nr_a3_d["nr_rsrp"]
            nr_a3_nr_arfcn= data_nr_a3_d["nr_arfcn"]


        if (event_info != 'null'):
            # 差分，识别event info 中的nr_pci  nr_rsrp
            nr_a3_d_pci = event_info
            nr_a3_d_rsrp = event_info
            nr_a3_d_earfcn =event_info
            data_nr_a3_new_nr_a3_d_earfcn  = before_one_second(ind, file_nr_d_pci_rsrp_earfcn, date) #10月30日，由于华为设备原因，event_info 中没有earfcn值，在前一秒内查找“NR Detected SSB NR-ARFCN  ”
            if  (data_nr_a3_new_nr_a3_d_earfcn["nr_d_arfcn"] != 'null'):
                nr_a3_d_earfcn = data_nr_a3_new_nr_a3_d_earfcn["nr_d_arfcn"]
                # print(type(nr_a3_d_earfcn))
                # print(nr_a3_d_earfcn)
            # nr_pci_b1=event_info.split(';')[1]
            # nr_rsrp_b1=event_info.split(';')[2]
        else:
            data_nr_a3=  before_one_second(ind, file_nr_d_pci_rsrp_earfcn,date)
            if (data_nr_a3["nr_d_pci"] != 'null')&(data_nr_a3["nr_d_arfcn"] != 'null')&(data_nr_a3["nr_d_rsrp"] != 'null'):
                nr_a3_d_pci = data_nr_a3["nr_d_pci"]
                nr_a3_d_rsrp = data_nr_a3["nr_d_rsrp"]
                nr_a3_d_earfcn = data_nr_a3["nr_d_arfcn"]

        ind_a3 =ind
        data_a3 = after_four_second(ind, file_event_name[file_event_name.event_name == 'LTEEventA3'],date)
        if (data_a3["date"] != 'null')&(data_a3["ind"] != 'null'):
            date_a3 = data_a3["date"]
            ind_a3 = data_a3["ind"]
        if(ind_a3 != ind):
            data_nr_a3_d = before_one_after_four_second (ind_a3 ,file_d_pci_rsrp_earfcn,date_a3)
            # data_nr_a3_d = before_one_second (ind_a3 ,file_d_pci_rsrp_earfcn,date_a3)
            if (data_nr_a3_d["date"] !='null')&(data_nr_a3_d["d_pci"] !='null')&(data_nr_a3_d["d_rsrp"] !='null')&(data_nr_a3_d["d_earfcn"] !='null'):
                d_pci = data_nr_a3_d["d_pci"]
                d_rsrp = data_nr_a3_d["d_rsrp"]
                d_earfcn =data_nr_a3_d["d_earfcn"]

        data_nrs_attempt = after_four_second(ind, file_event_name[file_event_name.event_name == 'NRSCellChangeAttempt'],date)
        if data_nrs_attempt["date"] != 'null':
            date_nrs_attempt = data_nrs_attempt["date"]

        ind_nrs_success = ind
        data_nrs_success = after_four_second(ind, file_event_name[file_event_name.event_name == 'NRSCellChangeSuccess'],date)
        if data_nrs_success["date"] != 'null':
            date_nrs_success = data_nrs_success["date"]
            ind_nrs_success = data_nrs_success["ind"]
        if (ind_nrs_success != ind):
            data_nr_pcc = after_four_second(ind_nrs_success, file_pci_rsrp_earfcn,date_nrs_success)
            if (data_nr_pcc["date"] != 'null') & (data_nr_pcc["pci"] != 'null') & (data_nr_pcc["rsrp"] != 'null') & (
                    data_nr_pcc["earfcn"] != 'null'):
                pcc_pci = data_nr_pcc["pci"]
                pcc_rsrp = data_nr_pcc["rsrp"]
                pcc_earfcn = data_nr_pcc["earfcn"]
            data_nr = after_four_second(ind_nrs_success, file_nr_pci_rsrp_earfcn,date_nrs_success)
            if (data_nr["date"] != 'null') & (data_nr["nr_pci"] != 'null') & (data_nr["nr_arfcn"] != 'null') & (
                    data_nr["nr_rsrp"] != 'null'):
                nr_pci = data_nr["nr_pci"]
                nr_earfcn = data_nr["nr_arfcn"]
                nr_rsrp = data_nr["nr_rsrp"]

        result0=result0.append({"index":ind , "type": "NREventA3", "lon":lon ,"la":la,
                                "date_nr_a3" :date_nr_a3 ,
                                "nr_a3_pci": nr_a3_pci , "nr_a3_earfcn":nr_a3_earfcn , "nr_a3_rsrp": nr_a3_rsrp ,
                                 "nr_a3_nr_rsrp":nr_a3_nr_rsrp , "nr_a3_nr_pci":nr_a3_nr_pci ,"nr_a3_nr_arfcn": nr_a3_nr_arfcn ,
                                "date_nr_a3_copy":date_nr_a3 ,
                                "nr_a3_d_pci": nr_a3_d_pci, "nr_a3_d_rsrp": nr_a3_d_rsrp , "nr_a3_d_earfcn":nr_a3_d_earfcn ,
                                "date_a3" : date_a3 ,
                                "d_pci":d_pci,"d_earfcn":d_earfcn,"d_rsrp":d_rsrp,
                                "date_nrs_attempt":date_nrs_attempt,"date_nrs_success":date_nrs_success,
                                "pcc_pci":pcc_pci,"pcc_rsrp":pcc_rsrp,"pcc_earfcn":pcc_earfcn,
                                "nr_pci":nr_pci,"nr_earfcn":nr_earfcn,"nr_rsrp":nr_rsrp},ignore_index=True)
        # print(result0)






# print(result0)

#添加人工判断逻辑





#PCellRSRP:-83;NCellPCI:212;NCellRSRP:-80
def chage_2_str_pci (str1) :
    if str1=='':
        return ""
    else:
        list = str1.split(";")
        str_PCI = list[1]
        # print(list)
        list_PCI = str_PCI.split(":")
        str_PCI_str = list_PCI[1]
        return  str_PCI_str
#512964;512964;512964

def chage_2_list_earfcn (str1) :
    if str1=='':
        earfcn_list = list()
    else:
        earfcn_list = str1.split(";")
        return  earfcn_list



ticks =time.asctime( time.localtime(time.time()) )
print(ticks)
print("输出优化方案")
##1 判断原因  (reason)     2 当前锚点小区  (now_pci_earfcn)   3  漏配NR小区频点、PCI  (omit_pci_earfcn)
result0['reason']= '正常'
result0['now_pci_earfcn']= ''
result0['omit_pci_earfcn']= ''
result0_sheet_1=  result0.shift(-1)

nr_a3_nr_pci_0 = ''
nr_a3_nr_arfcn_0=''
nr_a3_d_pci_0=''
nr_a3_d_earfcn_0=''
nr_pci_0=''
nr_earfcn_0=''


for row in  result0.itertuples():
    ind = getattr(row, 'Index')
    date_nrs_attempt =  getattr(row, 'date_nrs_attempt')
    date_nr_a3_copy = getattr(row, 'date_nr_a3_copy')
    date_a3 = getattr(row, 'date_a3')
    nr_a3_nr_pci = getattr(row, 'nr_a3_nr_pci')
    nr_a3_nr_arfcn = getattr(row, 'nr_a3_nr_arfcn')
    nr_a3_d_pci = getattr(row, 'nr_a3_d_pci')
    nr_a3_d_earfcn = getattr(row, 'nr_a3_d_earfcn')
    nr_pci = getattr(row, 'nr_pci')
    nr_earfcn = getattr(row, 'nr_earfcn')
    nr_pci=str(nr_pci)
    # print(nr_pci)
    if nr_pci=='nan':
       # print(nr_pci)
       nr_pci=''
    # print(nr_pci)
    nr_a3_d_pci = chage_2_str_pci(nr_a3_d_pci)
    nr_a3_d_list = chage_2_list_earfcn(nr_a3_d_earfcn)
    print(ind)
    # print("nr_a3_nr_pci_0==",nr_a3_nr_pci_0,"nr_a3_nr_arfcn_0==",nr_a3_nr_arfcn_0,"nr_a3_d_pci_0==",nr_a3_d_pci_0,"nr_a3_d_earfcn_0==",nr_a3_d_earfcn_0,"nr_pci_0==",nr_pci_0,"nr_earfcn_0==",nr_earfcn_0)
    # print("nr_a3_nr_pci==", nr_a3_nr_pci, "nr_a3_nr_arfcn==", nr_a3_nr_arfcn, "nr_a3_d_pci==", nr_a3_d_pci,"nr_a3_d_earfcn==",nr_a3_d_earfcn,"nr_pci==",nr_pci,"nr_earfcn==",nr_earfcn)
    if(nr_a3_nr_pci_0==nr_a3_nr_pci)&(nr_a3_nr_arfcn_0==nr_a3_nr_arfcn)&(nr_a3_d_pci_0==nr_a3_d_pci)&(nr_a3_d_earfcn_0==nr_a3_d_earfcn)&(nr_pci_0==nr_pci)&(nr_earfcn_0==nr_earfcn):
        # print("一样")
        nr_a3_nr_pci_0 = nr_a3_nr_pci
        nr_a3_nr_arfcn_0 = nr_a3_nr_arfcn
        nr_a3_d_pci_0 = nr_a3_d_pci
        nr_a3_d_earfcn_0 = nr_a3_d_earfcn
        nr_pci_0 = nr_pci
        nr_earfcn_0 = nr_earfcn
        result0.drop(index=ind,inplace=True)
        continue

    # print("不一样")
    nr_a3_nr_pci_0 = nr_a3_nr_pci
    nr_a3_nr_arfcn_0 = nr_a3_nr_arfcn
    nr_a3_d_pci_0 = nr_a3_d_pci
    nr_a3_d_earfcn_0 =nr_a3_d_earfcn
    nr_pci_0 = nr_pci
    nr_earfcn_0 = nr_earfcn

    if date_nrs_attempt =='' :
        flage_nr_attept_no = 0
        if date_nr_a3_copy != '':
            date_nr_a3_copy_format = pd.to_datetime(date_nr_a3_copy)
            date_a3_format = pd.to_datetime(date_a3)
            if (date_a3_format - date_nr_a3_copy_format).seconds < 2.5:
                result0.loc[ind, 'reason'] = '碰撞A3'
                flage_nr_attept_no = 1

        if flage_nr_attept_no == 0:
            result0.loc[ind, 'reason'] = '漏配5-5邻区'
            now_pci_earfcn = "PCI：" + str("{:.0f}".format(nr_a3_nr_pci)) + "，频点：" + str(
                "{:.0f}".format(nr_a3_nr_arfcn))
            result0.loc[ind, 'now_pci_earfcn'] = now_pci_earfcn
            omit_pci_earfcn = "PCI：" + nr_a3_d_pci+ "，频点：" + nr_a3_d_earfcn
            result0.loc[ind, 'omit_pci_earfcn'] = omit_pci_earfcn  # 需要拆分出pci  ,已完成差分pci  没有拆分 最强PCI  （√）

    else:
        if ( nr_pci==    nr_a3_nr_pci) & (nr_earfcn==nr_a3_nr_arfcn) :
            result0.loc[ind, 'reason'] = '无效'
        else:
            # N、P两列还没有拆分(√)
            if nr_pci !='':
                nr_pci=float(nr_pci)
                nr_pci =  str("{:.0f}".format(nr_pci))
            if nr_earfcn != '':
                nr_earfcn=float(nr_earfcn)
                nr_earfcn =  str("{:.0f}".format(nr_earfcn))
            # print('nr_pci ==' ,nr_pci, "  nr_a3_d_pci===  " ,nr_a3_d_pci  )
            # print("nr_earfcn===   " ,  nr_earfcn   ,"nr_a3_d_list:::"   ,nr_a3_d_earfcn)
            if  (nr_pci == nr_a3_d_pci) & (nr_earfcn in nr_a3_d_list):
                result0.loc[ind, 'reason'] = '正常'
            else:
                result0.loc[ind, 'reason'] = '漏配5-5最强邻区'
                now_pci_earfcn = "PCI：" + str("{:.0f}".format(nr_a3_nr_pci)) + "，频点：" + str(
                    "{:.0f}".format(nr_a3_nr_arfcn))
                result0.loc[ind, 'now_pci_earfcn'] = now_pci_earfcn
                omit_pci_earfcn = "PCI：" + nr_a3_d_pci + "，频点：" +   nr_a3_d_earfcn
                result0.loc[ind, 'omit_pci_earfcn'] = omit_pci_earfcn  # 需要拆分出pci  ,已完成差分pci  没有拆分 最强PCI  （√）




result0.columns=['序号','5-5工具切片类型','Longitude','Latitude',
                                '切片时间',
                                '变更前PCI','变更前频点','变更前电平',
                                '变更前NR电平','变更前NR PCI','变更前NR频点',
                                'NREventA3时间',
                                'NRA3携带最强小区PCI', 'NRA3携带最强小区电平','NR最强小区频点',
                                'EVENTA3时间',
                                'LTE最强电平小区PCI','LTE最强频点', 'LTE最强电平',
                                'NRSCellChangeAttemp时间','NRSCellChangeSuccess时间',
                                '变更后PCI','变更后电平','变更后频点',
                                '变更后NR -PCI','变更后NR-频点','变更后NR-电平',
                                '判断原因', '当前锚点小区', '漏配NR小区频点、PCI']


result0.to_csv(file_path_w, sep=',',encoding = 'gb2312', header=True, index=True)
ticks =time.asctime( time.localtime(time.time()) )
print(ticks)