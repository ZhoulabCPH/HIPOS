is_amp = True
import warnings

warnings.filterwarnings('ignore')
from torch.nn.functional import normalize
from dataset import *
from augmentation import *
import contrastive_loss
from my_variable_swin_v1 import *
from upernet import *
import matplotlib.pyplot as plt


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()





############################################################
####### Configuration
############################################################

class Net(nn.Module):

    def load_pretrain(self, ):
        print('loading %s ...' % self.arg.Model_Pretrained_Res)
        checkpoint = torch.load(self.arg.Model_Pretrained_Res, map_location=lambda storage, loc: storage)
        print(self.res.load_state_dict(checkpoint, strict=False))  # True

    def __init__(self, arg, resnet):
        super(Net, self).__init__()
        self.res = resnet
        self.arg = arg
        self.cluster_num = self.arg.Cluster_num
        self.feature_dim = self.arg.Feature_dim
        self.output_type = ['inference', 'loss']
        self.Downsampling = nn.AvgPool2d((7, 7), stride=None)
        self.Laten_Dim = self.arg.Instance_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.Laten_Dim, self.Laten_Dim),
            nn.ReLU(),
            nn.Linear(self.Laten_Dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.Laten_Dim, self.Laten_Dim),
            nn.ReLU(),
            nn.Linear(self.Laten_Dim, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self.bn = nn.BatchNorm1d(self.Laten_Dim, affine=False)
    def forward(self, batch):
        x_i = batch['image_Argument1']
        x_j = batch['image_Argument2']
        # Visual(x_i,x_j,batch)
        h_i = self.res(x_i)
        h_j = self.res(x_j)
        ##instance
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)
        ##cluster
        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)
        device = c_i.device
        criterion_instance = contrastive_loss.InstanceLoss(len(batch['index']), self.arg.instance_temperature,
                                                           device).to(device)
        criterion_cluster = contrastive_loss.ClusterLoss(self.cluster_num, self.arg.cluster_temperature, device).to(
            device)
        output = {}
        if 'loss' in self.output_type:
            output['loss_instance'] = criterion_instance(z_i, z_j)
            output['loss_cluster'] = criterion_cluster(c_i, c_j)
        if 'inference' in self.output_type:
            output['Cluste'] = torch.argmax(c_i, 1)
            output['probability_Ci'] = c_i
            output['probability_Cj'] = c_j
            output['instance_projector_i'] = z_i
            output['instance_projector_j'] = z_j
            output['res_i'] = h_i
            output['res_j'] = h_j
        return output
