import pandas as pd
import numpy as np
from dataset import *
import torch.optim
import warnings
warnings.filterwarnings('ignore')
from models import model
import torch.cuda.amp as amp
import argparse
from utils import yaml_config_hook, save_model
from sklearn.preprocessing import StandardScaler
std=StandardScaler()

def Clincial_Name(Clinical_Data):
    Clincial_Name=Clinical_Data["编码"].values
    Clincial_name_List=[]
    for name in Clincial_Name:
        Clincial_name_List.append(name.split('-')[1]+'-'+name.split('-')[2])
    return np.array(Clincial_name_List)
def Get_Pre_Model(Value,Datasets,net_):
    Test_Batch = {}
    Test_Batch['image'] = Value
    Test_Batch['name'] = Datasets.values[:, 0]
    net_.eval()
    net_.output_type = ['inference']
    with torch.no_grad():
        output = net_(Test_Batch)
    predict_test = output['Predict'].cpu().detach().numpy()
    predict_Score=output['Predict_score'].cpu().detach().numpy()[:,1]
    Datasets['Label'] = predict_test
    Datasets['predict_Score']=predict_Score
    return Datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    out_dir = args.Model_Out
    initial_checkpoint = args.initial_checkpoint
    start_lr = float(args.start_lr)
    batch_size = int(args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler(enabled=is_amp)
    net = model.Net(arg=args).to(device)
    if initial_checkpoint != 'None':
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_epoch = f['epoch']
        state_dict = f['state_dict']
        net.load_state_dict(state_dict, strict=False)  # True
    else:
        start_iteration = 0
        start_epoch = 0

    ##外部队列
    External = pd.read_csv(args.External)
    External_Value = torch.from_numpy(np.array(External.values[:, 1:16], dtype=np.float32)).cuda()
    Pre_External = Get_Pre_Model(External_Value, External, net)
    Pre_External.to_csv("Log/xx.csv")

