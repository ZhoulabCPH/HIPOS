is_amp = True
import warnings
warnings.filterwarnings('ignore')
from dataset import *
# 定义自编码器模型
class Net(nn.Module):
    def load_pretrain(self, ):
        print('loading %s ...' % self.arg.Model_Pretrained_Res)
        checkpoint = torch.load(self.arg.Model_Pretrained_Res, map_location=lambda storage, loc: storage)
        print(self.res.load_state_dict(checkpoint, strict=False))  # True
    def __init__(self, arg):
        super(Net, self).__init__()
        self.input_dim=arg.input_dim
        self.encoding_dim = arg.encoding_dim
        # 定义注意力层
        self.attention = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            nn.Tanh()
        )

        # 定义编码器
        self.encoder1 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim//2),
            nn.BatchNorm1d(self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim//2),
            nn.BatchNorm1d(self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU()
        )
        # 定义Laten层

        # 定义解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim//4),
            nn.BatchNorm1d(self.input_dim//4),
            nn.ReLU(),
            nn.Linear(self.input_dim//4, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, batch):
        x=batch['image']
        # x = batch
        # 计算注意力得分
        attention_scores = self.attention(x)
        # 使用注意力加权输入数据
        x_attention = x * attention_scores
        encoded1 = self.encoder1(x_attention)
        encoded2 = self.encoder2(x)

        encoded = torch.cat((encoded1, encoded2), dim=1)

        # 编码器过程
        decoded = self.decoder(encoded)
        # 定义交叉熵损失函数
        loss_function = nn.CrossEntropyLoss()
        MSE=nn.MSELoss()
        output={}
        if 'loss' in self.output_type:
            a=1
            output['MSELoss'] = MSE(encoded1, encoded2)
            output["loss_function"]=loss_function(decoded,batch['label'])
        if 'inference' in self.output_type:

            output['Predict'] =torch.argmax(decoded,1)
            output['Predict_score']=decoded
            output['attention_scores']=attention_scores
        return output


