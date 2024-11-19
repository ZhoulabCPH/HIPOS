import pandas as pd
import numpy as np
from scipy import stats
Patch_Feature=pd.read_csv("Patch_Feature_Train.csv").iloc[:,1:]
Patch_Cluster=pd.read_csv("Patch_Cluster_Train.csv").iloc[:,1:]
Label_Cluster=Patch_Cluster.groupby(by='Pre_Label')
Index_Feature=[]
Index_cluster=[]
for label_type,inf_cluster in Label_Cluster:
    cluster_inf = 'HPC:' + str(label_type)
    Index_cluster.append(cluster_inf)
    sorted_df = inf_cluster.sort_values(by=cluster_inf, ascending=False)
    # Index_Top = list(sorted_df.axes[0])
    Index_Top=list(sorted_df.axes[0])
    while len(Index_Top) < 100:
        # 在数组末尾添加适当数量的零填充
        Index_Top.append(Index_Top[0])
    datasetss=np.mean(Patch_Feature.values[Index_Top,1:-1],axis=0)
    Index_Feature.append(datasetss)

Index_array_=np.array(np.vstack(Index_Feature),dtype=np.float32)
Index_cluster=np.array(Index_cluster)
correlation_matrix = np.corrcoef(Index_array_, rowvar=True)
# correlation_matrix=stats.pearsonr(Index_Feature)
Corrletation_matrix_pd=pd.DataFrame(correlation_matrix,columns=Index_cluster,index=Index_cluster)
# Corrletation_matrix_pd.to_csv("Corrletation_matrix.csv")







