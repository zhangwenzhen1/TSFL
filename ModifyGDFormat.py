# encoding=utf-8
from Postgresql import *
import time
import pandas as pd
import numpy as np
class ModifyGDFormat(object):
    def __init__(self, df):
        self.df = df
    def chooseState(self, eptable):
        """
        筛选现网有业务工单
        :return:
        """
        df_all = pd.merge(self.df, eptable, on='CGI', how='left', suffixes=('', '_y'))  # pandas csv表左连接
        df = df_all.loc[df_all['state'] == '现网有业务']
        df.drop(columns=['state'], axis=1, inplace=True)
        df = df.reset_index(drop=False)
        df_all = df_all[['vccquestiontype', 'vcequestiontype', 'starttime', 'endtime', 'CGI', 'vcauxiliarypointer4',
                         'vcauxiliarypointer1', 'vcauxiliarypointer2', 'vcauxiliarypointer3', 'state',
                         'vcproblemtarget1']]
        df_all.columns = ['vccquestiontype', 'vcequestiontype', 'starttime', 'endtime', 'cgi', 'cellname', 'city',
                          'districtandcounty', 'vendor', 'state', 'vcproblemtarget']
        print("获取现网有业务工单成功")
        return df, df_all

    def SendTable(self, df, suoying):
        """
        修改为省派单式
        :param df: 要派发的工单数据
        :param b: 上次派单最大编号
        :return: send派发工单，v_return回单表工单
        """
        df['vcauxiliarypointer6'] = "JT" + (df.index + 1 + suoying).map(str)
        df['vcquestioncategory'] = "小区问题点"
        df['vcnetworksystem'] = "LTE"
        df['vcdatasource'] = "端到端信令分析优化"
        df['vccgi'] = "460-00-" + df['CGI'].map(str)
        df['vcspecialtag'] = "VOLTE两高两低-集团"
        df['vcproblemtarget3'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        df['vcproblemtarget5'] = "广东"
        df['vcproblemtarget2'] = "VOLTE-JTGD" + '-' + (
            time.strftime('%Y%m%d', time.localtime(time.time()))).format_map(str) \
                                 + '-' + df['vcauxiliarypointer6']
        send = df[
            ['vcquestioncategory', 'vccquestiontype', 'vcequestiontype', 'vcnetworksystem', 'starttime', 'endtime',
             'vcdatasource', 'vccgi', 'vcspecialtag', 'vcproblemtarget1', 'vcproblemtarget2', 'vcproblemtarget3',
             'vcproblemtarget5', 'vcauxiliarypointer1', 'vcauxiliarypointer2', 'vcauxiliarypointer3',
             'vcauxiliarypointer4', 'vcauxiliarypointer6']]
        send['vcauxiliarypointer1'] = np.where(
            (send['vcauxiliarypointer1'].isnull() | send['vcauxiliarypointer1'] == ''), '未识别地市',
            send['vcauxiliarypointer1'])
        lieming = ['vcequestiontype', 'vcnetworksystem', 'starttime', 'endtime', 'vcauxiliarypointer6']
        v_return = pd.DataFrame(columns=lieming)
        v_return['vcequestiontype'] = df['vcequestiontype']
        v_return['vcnetworksystem'] = df['vcnetworksystem']
        v_return['starttime'] = df['starttime']
        v_return['endtime'] = df['endtime']
        v_return['vcauxiliarypointer6'] = df['vcauxiliarypointer6']
        v_return = v_return[['vcequestiontype', 'vcnetworksystem', 'starttime', 'endtime', 'vcauxiliarypointer6']]
        print('工单格式修改完成')
        return send, v_return

    def Run(self):
        """
        连接数据库获取上次派单最大编号,并修改派单格式
        :return: send可派发工单，df_all 集团所有工单，v_return回单表需填字段
        """
        a = Postgresql()
        ##获取工参
        sql = "SELECT cgi,state,cityname,districtandcounty,vendor FROM volte.v_eptable"
        eptable = a.GetData(sql)
        # eptable = pd.DataFrame(eptable)
        eptable.columns = ['CGI', 'state', 'vcauxiliarypointer1', 'vcauxiliarypointer2', 'vcauxiliarypointer3']
        eptable = eptable.drop_duplicates('CGI')
        ##获取上次派单最大编号
        sql1 = "SELECT max(cast(replace(vcauxiliarypointer6,'JT','') as bigint)) as suoyin FROM volte.v_volte_send where vcauxiliarypointer6 like 'JT%' "
        suoying = a.GetData(sql1)
        # suoying = suoying[0][0]
        suoying = suoying.iat[0, 0]
        df, df_all = self.chooseState(eptable)
        send, v_return = self.SendTable(df, suoying)
        a.dateIntoPostgresql(send, 'volte.v_volte_send')
        print('派单表数据入库成功')
        a.dateIntoPostgresql(v_return, 'volte.v_volte_returnvaluation')
        print("回单表数据入库成功")
        a.dateIntoPostgresql(df_all,'volte.vn_volte_gdsend_jt')
        print("集团所有工单数据入库成功")
        a.finish()
        return send, v_return, df_all

if __name__ == "__main__":
    df = pd.read_csv('D:/集团工单/gd/paidan.csv', encoding='gbk')
    task = ModifyGDFormat(df)
    send, v_return, df_all = task.Run()
    print("集团下发派单数:{}".format(str(df_all.iloc[:, 0].size)))
    print("实际可派工单数:{}".format(str(send.iloc[:, 0].size)))

    # print(send.head())
