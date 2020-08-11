# encoding=utf-8
import pandas as pd
from Postgresql import *
import warnings

warnings.filterwarnings('ignore')


class JJtGdFK(object):
    def __init__(self, url, filename):
        self.url = url
        self.filename = filename

    def ReadFile(self, sheetname, usecols):
        """
        读取文件，筛选广东问题小区，修改列名
        :param sheetname:
        :param usecols:
        :return:
        """
        df = pd.read_excel(self.url + '/' + self.filename, sheet_name=sheetname, usecols=usecols)
        df = df.loc[df['省份'] == '广东']
        return df

    def getBackTablekpi(self, sheetname):

        if sheetname == '低接入小区闭环管理反馈表':
            usecols = ['省份', 'CGI', 'VoLTE话务量', 'VoLTE接通率']
            df = self.ReadFile(sheetname, usecols)
            df.columns = ['province', 'cgi', 'voltetraval', 'volteradconnratio']
            df['problemtype'] = 'VOLTEJTD'
            print("低接入反馈指标获取成功")
        elif sheetname == '高掉话小区闭环管理反馈表':
            usecols = ['省份', 'CGI', '话务量', '掉话率']
            df = self.ReadFile(sheetname, usecols)
            df.columns = ['province', 'cgi', 'voltetraval', 'volteraddropratio']
            df['problemtype'] = 'VOLTERABDXG'
            print("高掉话反馈指标获取成功")
        elif sheetname == '低SRVCC无线切换成功率小区闭环管理反馈表':
            usecols = ['省份', 'cgi', 'LTE到2G切换失败次数', 'SRVCC无线切换成功率']
            df = self.ReadFile(sheetname, usecols)
            df.columns = ['province', 'cgi', 'failoutgeran', 'srvcchosucratio']
            df['problemtype'] = 'ESRVCCQHC'
            print("低SRVCC无线切换反馈指标获取成功")
        elif sheetname == '上行高丢包小区闭环管理反馈表':
            usecols = ['省份', 'CGI', '上行平均丢包率']
            df = self.ReadFile(sheetname, usecols)
            df.columns = ['province', 'cgi', 'uppdcplossratio']
            df['problemtype'] = 'VOLTESXGDB'
            print("上行高丢包反馈指标获取成功")
        elif sheetname == '下行高丢包小区闭环管理反馈表':
            usecols = ['省份', 'CGI', '下行平均丢包率']
            df = self.ReadFile(sheetname, usecols)
            df.columns = ['province', 'cgi', 'downpdcplossratio']
            df['problemtype'] = 'VOLTEXXGDB'
            print("下行高丢包反馈指标获取成功")
        elif sheetname == '上行高吞字小区反馈表':
            usecols = ['省份', '小区cgi', '上行高吞字采样点占比']
            df = self.ReadFile(sheetname, usecols)
            df.columns = ['province', 'cgi', 'uptzsamprate']
            df['problemtype'] = 'VOLTESXGTZ'
            print("上行高呑子反馈指标获取成功")
        elif sheetname == '下行高吞字小区反馈表':
            usecols = ['省份', '小区cgi', '下行高吞字采样点占比']
            df = self.ReadFile(sheetname, usecols)
            df.columns = ['province', 'cgi', 'downtzsamprate']
            df['problemtype'] = 'VOLTEXXGTZ'
            print("下行高呑子反馈指标获取成功")
        return df

    def Go_run(self):
        VOLTEJTD = self.getBackTablekpi('低接入小区闭环管理反馈表')
        VOLTERABDXG = self.getBackTablekpi('高掉话小区闭环管理反馈表')
        ESRVCCQHC = self.getBackTablekpi('低SRVCC无线切换成功率小区闭环管理反馈表')
        VOLTESXGDB = self.getBackTablekpi('上行高丢包小区闭环管理反馈表')
        VOLTEXXGDB = self.getBackTablekpi('下行高丢包小区闭环管理反馈表')
        VOLTESXGTZ = self.getBackTablekpi('上行高吞字小区反馈表')
        VOLTEXXGTZ = self.getBackTablekpi('下行高吞字小区反馈表')
        result = pd.concat([VOLTERABDXG, VOLTEJTD, ESRVCCQHC, VOLTESXGDB, VOLTEXXGDB, VOLTESXGTZ, VOLTEXXGTZ], axis=0,
                           ignore_index=True)
        a = Postgresql()
        sql = "SELECT cgi,cityname FROM volte.v_eptable "
        eptable = a.GetData(sql)
        eptable = pd.DataFrame(eptable)
        eptable.columns = ['cgi', 'city']
        eptable = eptable.drop_duplicates('cgi')
        result = pd.merge(result, eptable, on='cgi', how='left', suffixes=('', '_y'))  # pandas csv表左连接
        result['flag'] = 1
        print("生成集团“两高两低”反馈表成功")
        return result


if __name__ == "__main__":
    path = r'D:/集团工单/gd'  # 指定存放文件的地址
    path2 = r'D:/集团工单'  # 指定存放结果文件
    # filename = input("请输入您要读取的文件名：")
    filename = '两低两高小区问题跟踪表20200501.xlsx'
    a = JJtGdFK(path, filename)
    df = a.Go_run()
    print(df.columns)
    df.to_csv(path2 + '/fankui.csv', header=1, encoding='gbk', index=False)  # 保存列名存储
