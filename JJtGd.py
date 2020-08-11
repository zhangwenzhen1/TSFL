# encoding=utf-8
import datetime
from Postgresql import *
class JJtGd(object):
    def __init__(self, url, filename):
        self.url = url
        self.filename = filename

    def ReadFile(self, sheetname, usecols=None):
        """
        读取文件，筛选广东问题小区，修改列名
        :param sheetname:
        :param usecols:
        :return:
        """
        df = pd.read_excel(self.url + '/' + self.filename, sheet_name=sheetname, usecols=usecols)
        df = df.loc[df['省份'] == '广东']
        df.rename(columns={'小区名称': 'vcauxiliarypointer4'}, inplace=True)
        return df, sheetname

    def GetWorstCellKpi(self, df, sheetname):

        if sheetname == '低接入小区明细参考表':
            df = df.sort_values(by=['CGI', '无线接通率(QCI1)'], ascending=[True, True])
            df = df.drop_duplicates(['CGI'], keep='first')
            df['vccquestiontype'] = 'VoLTE无线接通差小区(语音)-19'
            df['vcequestiontype'] = 'VOLTEJTD'
            df['无线接通率(QCI1)'] = df['无线接通率(QCI1)'].apply(lambda x: format(x, '.2%'))
            df['vcproblemtarget1'] = "[无线接通率(QCI1):]" + df['无线接通率(QCI1)'] + "[VoLTE话务量]:" + \
                                     df['VoLTE语音话务量'].map(str)
            print("低接入指标获取成功")
        elif sheetname == '低SRVCC无线切换成功率小区明细参考表':
            df.rename(columns={'时间': '日期'}, inplace=True)
            df = df.sort_values(by=['CGI', 'SRVCC无线切换成功率'], ascending=[True, True])
            df = df.drop_duplicates(['CGI'], keep='first')
            df['vccquestiontype'] = 'eSRVCC切换差小区-19'
            df['vcequestiontype'] = 'ESRVCCQHC'
            df['SRVCC无线切换成功率'] = df['SRVCC无线切换成功率'].apply(lambda x: format(x, '.2%'))
            df['vcproblemtarget1'] = "[eSRVCC切换成功率]" + df['SRVCC无线切换成功率'] + "[eSRVCC切换失败次数]" + \
                                     df['SRVCC无线切换失败次数'].map(str)
            print("低SRVCC无线切换指标获取成功")
        elif sheetname == '高掉话小区明细参考表':
            df = df.sort_values(by=['CGI', 'E-RAB掉线率(QCI1)(小区级)'], ascending=[True, False])
            df = df.drop_duplicates(['CGI'], keep='first')
            df['vccquestiontype'] = 'VoLTE E-RAB掉线高小区(语音)-19'
            df['vcequestiontype'] = 'VOLTERABDXG'
            df['E-RAB掉线率(QCI1)(小区级)'] = df['E-RAB掉线率(QCI1)(小区级)'].apply(
                lambda x: format(x, '.2%'))
            df['vcproblemtarget1'] = "[E-RAB掉线率(QCI1)(小区级)]:" + df['E-RAB掉线率(QCI1)(小区级)'] + "[VoLTE话务量]:" + \
                                     df['VoLTE语音话务量'].map(str)
            print("高掉话指标获取成功")
        elif sheetname == '上行高丢包小区明细参考表':
            df = df.sort_values(by=['CGI', '上行平均丢包率'], ascending=[True, False])
            df = df.drop_duplicates(['CGI'], keep='first')
            df['vccquestiontype'] = 'VoLTE上行高丢包小区-19'
            df['vcequestiontype'] = 'VOLTESXGDB'
            df['上行平均丢包率'] = df['上行平均丢包率'].apply(lambda x: format(x, '.2%'))
            df['vcproblemtarget1'] = "[上行丢包率]:" + df['上行平均丢包率'] + "[上行丢包率采样点点数]:" + \
                                     df['上行总样本数'].map(str)
            print("上行高丢包指标获取成功")
        elif sheetname == '下行高丢包小区明细参考表':
            df.rename(columns={'时间': '日期'}, inplace=True)
            df = df.sort_values(by=['CGI', '下行平均丢包率'], ascending=[True, False])
            df = df.drop_duplicates(['CGI'], keep='first')
            df['vccquestiontype'] = 'VoLTE下行高丢包小区-19'
            df['vcequestiontype'] = 'VOLTEXXGDB'
            df['下行平均丢包率'] = df['下行平均丢包率'].apply(lambda x: format(x, '.2%'))
            df['vcproblemtarget1'] = "[下行丢包率]:" + df['下行平均丢包率'] + "[下行丢包率采样点点数]:" + \
                                     df['下行总样本数'].map(str)
            print("下行高丢包指标获取成功")
        elif sheetname == '上行高吞字小区明细参考表':
            df.rename(columns={'小区cgi': 'CGI'}, inplace=True)
            df = df.sort_values(by=['CGI', '上行高吞字采样点占比'], ascending=[True, False])
            df = df.drop_duplicates(['CGI'], keep='first')
            df['vccquestiontype'] = 'VoLTE上行高吞字小区-19'
            df['vcequestiontype'] = 'VOLTESXGTZ'
            df['上行高吞字采样点占比'] = df['上行高吞字采样点占比'].apply(lambda x: format(x, '.2%'))
            df['vcproblemtarget1'] = "[上行高吞字率]:" + df['上行高吞字采样点占比'] + "[上行丢包率采样点点数]:" + \
                                     df['上行总采样点数'].map(str)
            print("上行高呑子指标获取成功")
        elif sheetname == '下行高吞字小区明细参考表':
            df.rename(columns={'小区cgi': 'CGI'}, inplace=True)
            df = df.sort_values(by=['CGI', '下行高吞字采样点占比'], ascending=[True, False])
            df = df.drop_duplicates(['CGI'], keep='first')
            df['vccquestiontype'] = 'VoLTE下行高吞字小区-19'
            df['vcequestiontype'] = 'VOLTEXXGTZ'
            df['下行高吞字采样点占比'] = df['下行高吞字采样点占比'].apply(lambda x: format(x, '.2%'))
            df['vcproblemtarget1'] = "[下行高吞字率]:" + df['下行高吞字采样点占比'] + "[下行丢包率采样点点数]:" + \
                                     df['下行总采样点数'].map(str)
            print("下行高呑子指标获取成功")
        df = df[['日期', 'vccquestiontype', 'vcequestiontype', 'CGI', 'vcproblemtarget1', 'vcauxiliarypointer4']]
        if df.iloc[:, 0].size >= 1:
            aa, bb = self.GetDate(df)
            df['starttime'] = bb
            df['endtime'] = aa
        # df.drop(['日期'], axis=1, inplace=True)
        return df

    def GetDate(self, a):
        """
        获取工单生成日期
        :param a:
        :return:
        """
        a['日期'] = pd.to_datetime(a['日期'], format='%Y%m%d')
        aa = a['日期'].max()
        aa = (aa + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        bb = a['日期'].min()
        return aa, bb

    def Go(self, sheetname):
        df, sheetname = self.ReadFile(sheetname)
        df = self.GetWorstCellKpi(df, sheetname)
        return df

    def Run(self):
        df_VOLTEJTD = self.Go('低接入小区明细参考表')
        df_VOLTERABDXG = self.Go('高掉话小区明细参考表')
        df_ESRVCCQHC = self.Go('低SRVCC无线切换成功率小区明细参考表')
        df_VOLTESXGDB = self.Go('上行高丢包小区明细参考表')
        df_VOLTEXXGDB = self.Go('下行高丢包小区明细参考表')
        df_VOLTESXGTZ = self.Go('上行高吞字小区明细参考表')
        df_VOLTEXXGTZ = self.Go('下行高吞字小区明细参考表')
        df = df_VOLTEJTD.append(df_VOLTERABDXG).append(df_ESRVCCQHC).append(df_VOLTESXGDB).append(df_VOLTEXXGDB).append(
            df_VOLTESXGTZ).append(df_VOLTEXXGTZ)
        df = df.reset_index(drop=False)
        df.drop(['index'], axis=1, inplace=True)
        return df

