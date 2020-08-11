import pandas as pd
import numpy as np
df = pd.read_excel("D:\临时文件/0_故障小区识别结果(1).xlsx",sheet_name='Sheet3')
print(df.head())
M_ALARM = pd.read_csv("D:\临时文件/M_ALARM_EUTRANCELL0708.csv", encoding='utf-8')
ALARM_A = pd.read_csv("D:\临时文件/告警_爱立信.csv", encoding='gbk')

print(ALARM_A.columns)
ALARM_D = pd.read_csv("D:\临时文件/告警_大唐.csv", encoding='gbk')
print(ALARM_D.columns)
ALARM_H = pd.read_csv("D:\临时文件/告警_华为.csv", encoding='gbk')
print(ALARM_H.columns)
ALARM_N = pd.read_csv("D:\临时文件/告警_诺基亚.csv", encoding='gbk')
print(ALARM_N.columns)
ALARM_Z = pd.read_csv("D:\临时文件/告警_中兴.csv", encoding='gbk')
print(ALARM_Z.columns)
ALARM = ALARM_A.append(ALARM_D ).append(ALARM_H).append(ALARM_N).append(ALARM_Z)
ALARM.to_csv('D:\临时文件/故障1.csv', encoding='gbk',index=False)
# RESULT = pd.merge(df,M_ALARM,on='cgi', how ='outer', suffixes=('', '_y'))
# RESULT["告警是否影响业务"] = np.where((RESULT['standard_alarm_id'].notnull() &
#                                RESULT['standard_alarm_id'].isin(ALARM['网管告警ID'])),'是','')
# RESULT.to_csv('D:\临时文件/故障.csv', encoding='gbk',index=False)