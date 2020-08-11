# encoding=utf-8
import psycopg2
from io import StringIO
import pandas as pd
class Postgresql(object):
    # __slots__ = ('database', 'conn','user','password','host','port')
    def __init__(self):
        '''
        初始化数据库名，用户，密码，地址，端口
        :param database: 数据库名
        :param user: 用户
        :param password: 密码
        :param host: 地址
        :param port: 端口
        '''
        self.database = '**'
        self.user = '**'
        self.password = '**'
        self.host = '**'
        self.port = '**'
        self.conn = psycopg2.connect(database=self.database, user=self.user, password=self.password, host=self.host,
                                     port=self.port)

    def dateIntoPostgresql(self, a, b):
        """
        csv数据入postgresql库
        :param a: 要入的数据
        :param b: 入库的表名
        :return:
        """
        col = a.columns
        # dataframe类型转换为IO缓冲区中的str类型
        output = StringIO()
        a.to_csv(output, sep='\t', index=False, header=False)
        output1 = output.getvalue()
        conn = self.conn
        cur = conn.cursor()
        cur.copy_from(StringIO(output1), b, null='', columns=col)
        cur.close()

    def GetData(self, sql):
        """
        获取查询数据
        :param sql: 查询数据sql
        :return date: 查询数据
        """
        conn = self.conn
        cur = conn.cursor()
        cur.execute(sql)
        date = cur.fetchall()
        date = pd.DataFrame(date)
        cols = cur.description
        # 将数据truple转换为DataFrame
        col = []
        for i in cols:
            col.append(i[0])
        # print(col)
        date.columns = col
        cur.close()
        return date

    def finish(self):
        self.conn.commit()
        self.conn.close()

if __name__ == "__main__":
    a = Postgresql()
    # df = pd.read_csv('D:\集团工单/back.csv', encoding='gbk')
    # df1=pd.read_csv('D:\集团工单/riqi1.csv', encoding='gbk')
    # # a.dateIntoPostgresql(df, "volte.vn_gdcellkpi_group")
    # a.dateIntoPostgresql(df1, "volte.vn_gd_group_date")
    # sql ="SELECT * from volte.userinfor where s_date <'2020-06-11 00:00:00' "
    # sql = "SELECT cgi,state,cityname,districtandcounty,vendor FROM volte.v_eptable"

    # sql1 = "SELECT max(cast(replace(vcauxiliarypointer6,'JT','') as bigint)) as suoyin FROM volte.v_volte_send where vcauxiliarypointer6 like 'JT%' "
    sql2 ="SELECT * FROM volte.vn_xdrresultscore xdr LEFT JOIN volte.vn_complain_reason reason ON xdr.request_no = reason.request_no WHERE TO_CHAR(xdr.accept_time, 'yyyy-MM-dd') = '2020-07-27'"
    date = a.GetData(sql2)
    
    # print('suoy'+str(date.iat[0,0]))
    date = date.drop_duplicates('request_no')
    print(date.iloc[:, 0].size)
    # date = pd.DataFrame(date)
    # date.columns = ['CGI', 'state', 'vcauxiliarypointer1', 'vcauxiliarypointer2', 'vcauxiliarypointer3']
    # print(date.head())
    date.to_csv('D:\集团工单/USETOUSU1.csv', index= False,encoding='gbk')
    a.finish()