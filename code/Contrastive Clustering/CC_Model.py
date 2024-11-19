import numpy as np
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from dataset import *
import warnings

warnings.filterwarnings('ignore')
from models import resnet,model
import math
import torch.cuda.amp as amp
import torchvision
import argparse
from utils import yaml_config_hook, save_model
import tables
def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']
############################################################
####### Validation
############################################################
def initialization():
    Y_Sample_Name_List=np.array(0,dtype=np.str_)
    Y_Weight_List=np.array([0]).reshape(-1,1)
    Y_Pre_List = np.array([0]).reshape(-1, 1)
    Y_True_List = np.array([0]).reshape(-1, 1)
    Instance=np.zeros((1,128),dtype=np.float32)
    Cluster_F = np.zeros((1, 64), dtype=np.float32)

    return Y_Sample_Name_List,Y_Weight_List,Y_Pre_List,Y_True_List,Instance,Cluster_F
def Get_Evaluation_Metrics(output,batch):
    Weight = np.array(output['Cluste'].detach().cpu())
    # y_Pre = np.array(output['probability'].detach().cpu())
    y_name=np.array(batch['name'])
    return Weight,y_name
def Clincial_Name(Clinical_Data):
    Clincial_Name=Clinical_Data["编码"].values

    Clincial_name_List=[]
    for name in Clincial_Name:
        Clincial_name_List.append(name.split('-')[1]+'-'+name.split('-')[2])
    return np.array(Clincial_name_List)

def Get_Patient_clusterRatio_External():

    ##研究一下片子的构造
    ExternalHYD=pd.read_csv("Log/Result/External/Patch_Level_ExternalHYD.csv")
    Data_=ExternalHYD

    Patient_Name = np.array([name.split("_")[0] for name in Data_['Name'].values])
    Data_["Patient"] = Patient_Name
    Patient_Datasets = []
    group_data = Data_.groupby(by="Patient")
    for pd_data in group_data:
        Clusterfile = np.zeros((1, 63 + 1), dtype=np.float32)
        for clusters in pd_data[1]["Pre_Label"].values:
            Clusterfile[0, int(clusters)] += 1
        Clusterfile = Clusterfile / np.sum(Clusterfile)

        patient_name = np.array(pd_data[0]).reshape(-1, 1)
        Iter_datasets = np.hstack((patient_name, Clusterfile)).squeeze()
        Patient_Datasets.append(Iter_datasets)

    Index_cluster = ['Cluster:{}'.format(i) for i in range(63 + 1)]
    Index = np.hstack((np.array(["Sample_Name"]), np.array(Index_cluster)))
    My_Pd = pd.DataFrame(Patient_Datasets, columns=Index)
    My_Pd['Sample_Name'] = np.array(My_Pd['Sample_Name'].values, dtype=np.int64)

    ##Clincial_data
    Clincial = pd.read_csv("../../Datasets/Clincial_datasets/使用的临床表格/"
                           "Clincial_ExternalHYD.csv")
    Need_Clincial = Clincial[["Sample_Name", 'DFSState', 'DFS', 'OSState', 'OS']]
    Samples_Name = pd.merge(My_Pd, Need_Clincial, how='inner', on='Sample_Name')

    # df['Sample_Name'].to_csv("Patient.csv")
    return Samples_Name
def Get_Patient_ClsterRatio(data_Orial,Max_Dim=63):
    Data_ = data_Orial
    Patient_Name=np.array([name.split("_")[0] for name in data_Orial['Name'].values])
    Data_["Patient"]=Patient_Name
    Patient_Datasets=[]
    group_data=Data_.groupby(by="Patient")
    for pd_data in group_data:
        Clusterfile = np.zeros((1, Max_Dim+1), dtype=np.float32)
        for clusters in pd_data[1]["Pre_Label"].values:
            Clusterfile[0,int(clusters)]+=1
        Clusterfile=Clusterfile/np.sum(Clusterfile)
        patient_name=np.array(pd_data[0]).reshape(-1,1)
        Iter_datasets=np.hstack((patient_name,Clusterfile)).squeeze()
        Patient_Datasets.append(Iter_datasets)


    Index_cluster = ['Cluster:{}'.format(i) for i in range(Max_Dim+1)]
    Index = np.hstack((np.array(["Sample_Name"]), np.array(Index_cluster)))
    My_Pd=pd.DataFrame(Patient_Datasets,columns=Index)
    ##添加临床信息
    Orial_Feature = ['Sample_Name','OS', 'OSState', 'DFS','DFSState']

    Clinical_Data=pd.read_csv('../../Datasets/Clincial_datasets/使用的临床表格'
                                '/Clincial_ExternalTZ.csv')[Orial_Feature]
    My_Pds=pd.merge(My_Pd,Clinical_Data,how='inner',on='Sample_Name')




    return My_Pds

if __name__ == '__main__':
    ##读取Config配置
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    ##读取配置中的模型参数
    initial_checkpoint = args.initial_checkpoint
    batch_size = int(args.batch_size)

    #周孙课题组科研结果储存

    ##读取数据集
    '''
    1.读取图片的名字.csv  Train_Patient
    2.读取图片本身.H5D    H5DTrain_Image
    '''

    Train_Patient="xx.csv"
    train_df = pd.read_csv(Train_Patient)
    Filename=initial_checkpoint.split("/")[-1].split('.')[0]
    H5DTrain_Image="xx.hdf5"
    Store_File = tables.open_file(args.H5DTrain_Image, mode='r')
    Train_patches = Store_File.root.patches
    train_dataset = CreatDataset(train_df,Train_patches, Transform_)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=int(args.workers),  # if dubug 0
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate)

    ##读取模型参数
    res = resnet.get_resnet("ResNet50")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler(enabled=is_amp)
    net = model.Net(arg=args,resnet=res).to(device)
    if initial_checkpoint != 'None':
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_epoch = f['epoch']
        state_dict = f['state_dict']
        net.load_state_dict(state_dict, strict=False)  # True
    else:
        start_iteration = 0
        start_epoch = 0

    ##模型的验证
    Train_Name_List, Train_Weight_List, Train_Pre_List, Train_True_List,Instance,Cluster_F = initialization()
    net = net.eval()
    for t, batch in enumerate(train_loader):
        if t%100==0:
            print("Train—进度:{}%".format(t*100/len(train_loader)))
        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            batch_size = len(batch['index'])
            batch['image_Argument1'] = batch['image_Argument1'].cuda()
            batch['image_Argument2'] = batch['image_Argument2'].cuda()
            output = net(batch)
            PreDice=output['Cluste'].cpu().detach().numpy()
            InstanceFeature=output['instance_projector_i'].cpu().detach().numpy()
            ClusterFeature=output['probability_Ci'].cpu().detach().numpy()
            ##构造输出文件
            y_Pre, y_name = Get_Evaluation_Metrics(output,batch)
        Train_Pre_List = np.hstack((Train_Pre_List, np.array(y_Pre).reshape(1,-1)))

        Cluster_F = np.vstack((Cluster_F, ClusterFeature))
        Instance=np.vstack((Instance,InstanceFeature))
        Train_Name_List = np.hstack((Train_Name_List, y_name))

    ##将模型预测的数据输出为csv文件
    Cluster_Con=["Name"] +["Cluster:{}".format(i) for i in range(args.Cluster_num)]+['Pre_Label']
    Cluster_Con_Features=np.hstack((Train_Name_List[1:].reshape(-1,1),Cluster_F[1:,:],Train_Pre_List[:,1:].reshape(-1,1)))
    Datasets_Cluster = pd.DataFrame(Cluster_Con_Features, columns=Cluster_Con)
    Datasets_Cluster.to_csv("Log/xx.csv")

