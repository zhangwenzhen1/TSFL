import pandas as pd
import os
def csv_merge(path):
    # 返回.csv格式所有文件名的列表list
    file_name_list = os.listdir(path)
    #for循环获取所有.csv格式文件的绝对地址的列表list
    file_dir_list=[os.path.join(path,x) for x in file_name_list]
    print(file_dir_list)
    print(len(file_dir_list))
    #定义DataFrame类型的变量df用来存放获取的所有数据
    df = pd.DataFrame()
    j=0
    #for循环逐个读取每个csv里面的数据
    for i in file_name_list:
        if j in range(0,len(file_dir_list)):
            if (i[-3:] == 'csv'):  # 筛选只读取csv结尾的文件
                # read_csv方法，参数sheet_name表示读取的工作簿，skiprows表示忽略几行，usecols表示读取指定的列
                csv1 = pd.read_csv(file_dir_list[j])
                # concat方法合并多个文件的数据
                df = pd.concat([df, csv1], ignore_index=True)
                j=j+1
    print(df.iloc[:,0].size)
    return  df