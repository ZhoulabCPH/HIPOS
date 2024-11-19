import pandas as pd
import numpy as np

##读取第一步生成的Cluster文件
Patch_Level=pd.read_csv("xx.csv")[['Name','Pre_Label']]

##使用固定文件，构造簇类
Inter_Cluster_Corr=pd.read_csv("xx.csv")
replace_dict = dict(zip(Inter_Cluster_Corr['Cluster'], Inter_Cluster_Corr['HIPO']))
Patch_Level['HIPO'] = Patch_Level.iloc[:, -1].replace(replace_dict)
Patch_Level['Patient']=[name.split('_')[0] for name in Patch_Level['Name']]
##构造patient——level特征
count_df = Patch_Level.groupby(['Patient', 'HIPO']).size().unstack(fill_value=0)
proportion_df = count_df.div(count_df.sum(axis=1), axis=0)
proportion_df.columns = [f'HIPO{i}' for i in proportion_df.columns]
Patient_Level=proportion_df


Patient_Level.to_csv("./Log/xx.csv")







Patient_Level=1





