import pandas as pd
import numpy as np
from dataset import *
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,roc_curve, auc, confusion_matrix
import torch.optim
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import math
from models import attention_model
import torch.cuda.amp as amp
import argparse
from utils import yaml_config_hook, save_model
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']
def Clincial_Name(Clinical_Data):
    Clincial_Name=Clinical_Data["编码"].values
    Clincial_name_List=[]
    for name in Clincial_Name:
        Clincial_name_List.append(name.split('-')[1]+'-'+name.split('-')[2])
    return np.array(Clincial_name_List)
def Get_Pre_Model(Value,Datasets,net_,choice):
    net_.eval()
    net_.output_type = ['inference']
    with torch.no_grad():
        output = net_(Value)
    Predict= torch.argmax(output, 1)
    predict_test = Predict.cpu().detach().numpy()
    # predict_Score=output['Predict_score'].cpu().detach().numpy()[:,1]
    if choice=="Discovery":
        Surival_Data = Datasets[['Sample_Name', 'OS', 'OSState', 'DFS', 'DFSState','Label']]
    else:
        Surival_Data = Datasets[['Sample_Name', 'OS', 'OSState', 'DFS', 'DFSState']]
    Surival_Data['predict_Label'] = predict_test
    Surival_Data['predict_Score']=output[:,1].cpu().detach().numpy()
    return Surival_Data
def Result_KMFile(Input):
    scores = Input['attention_scores'].cpu().detach().numpy().squeeze()
    indices = np.argsort(scores)
    probability = Input['probability'].cpu().detach().numpy()
    result = np.empty((probability.shape[0], probability.shape[1], args.Tile_num))
    for i in range(probability.shape[0]):
        result[i] = np.transpose(probability[i, :, indices[i]])
    WSI_Feature = np.sum(result, axis=2)
    return WSI_Feature
def L1Regulation(net,lambda1):
    L1_reg = 0
    loss=0
    for param in net.parameters():
        L1_reg += torch.sum(torch.abs(param))
    loss += lambda1 * L1_reg
    return loss
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

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler(enabled=is_amp)
    net = attention_model.Net(arg=args).to('cpu')
    if initial_checkpoint != 'None':
        a=1
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_epoch = f['epoch']
        state_dict = f['state_dict']
        net.load_state_dict(state_dict, strict=False)  # True
    else:
        start_iteration = 0
        start_epoch = 0

    ##对发现集合进行推理
    Train=pd.read_csv("xx.csv").iloc[:,1:]
    Train['Truelabel']= Train['Label'].values-1
    Test = pd.read_csv("xx.csv").iloc[:,1:]
    Test['Truelabel'] = Test['Label'].values - 1
    External_BJ=pd.read_csv("xx.csv")

    datasets_discovery = pd.concat([Train, Test], axis=0).reset_index().iloc[:, 1:]

    datasets_discovery = Train

    Feature_Values=np.array(datasets_discovery.values[:, 1:16], dtype=np.float32)
    Train_Value = torch.from_numpy(Feature_Values)
    Pre_Train = Get_Pre_Model(Train_Value, datasets_discovery, net,'Discovery')
    true_labels=Pre_Train['Label'].values-1
    predictions=Pre_Train['predict_Score'].values
    # 计算ROC曲线的参数
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    ##评估的其他指标：
    ACC= metrics.accuracy_score(true_labels, Pre_Train['predict_Label'])
    AUC=metrics.roc_auc_score(true_labels, Pre_Train['predict_Label'])
    AUPR=metrics.average_precision_score(true_labels, Pre_Train['predict_Label'])
    F1SCORE=metrics.f1_score(true_labels, Pre_Train['predict_Label'])







    ##使用SHAP进行可视化模型
    net.output_type = ['inference']
    net.eval()
    scaled_data = Train_Value.to('cpu')
    explainer = shap.DeepExplainer(net,scaled_data)  # Use a small subset of samples for explanation
    shap_values = explainer.shap_values(scaled_data,check_additivity=False)


    ####################全局可解释性与局部可解释性一起放在了一张图上####################

    if isinstance(shap_values, list):
        shap_values_numpy = [value for value in shap_values]  # 多类别情况，直接使用数组
    else:
        shap_values_numpy = shap_values  # 单类别情况，直接使用数组

    # 输出 Bar 图
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, scaled_data, plot_type="bar", show=False)
    # plt.savefig("figure_2.pdf")
    plt.show()
    shap.summary_plot(shap_values[1], scaled_data, plot_type="dot")
    plt.show()

    # 额外图 1: Dot 图，添加特征名
    plt.ioff()  # 关闭交互模式
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.summary_plot(
        shap_values_numpy[0] if isinstance(shap_values_numpy, list) else shap_values_numpy,
        scaled_data,
        feature_names=[f"Feature {i}" for i in range(scaled_data.shape[1])],
        plot_type="dot"
    )
    plt.savefig("SHAP_numpy_summary_plot.pdf", format='pdf', bbox_inches='tight')
    # plt.show()



    shap.TreeExplainer(net,scaled_data)
    explainer = shap.TreeExplainer(net)  # 计算shap值为numpy.array数组
    shap_values_numpy = explainer.shap_values(scaled_data,check_additivity=False)
    shap_values_numpy

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values_numpy, scaled_data, feature_names=X.columns, plot_type="dot", show=False)
    plt.savefig("SHAP_numpy summary_plot.pdf", format='pdf', bbox_inches='tight')


    import shap
    import matplotlib.pyplot as plt

    # Step 1: 使用 SHAP 生成解释性数据
    net.output_type = ['inference']
    net.eval()
    scaled_data = Train_Value.to('cpu')

    # 初始化 SHAP DeepExplainer
    explainer = shap.DeepExplainer(net, scaled_data)
    shap_values = explainer.shap_values(scaled_data, check_additivity=False)

    # 将shap_values转换为numpy格式，方便之后使用
    shap_values_numpy = [value.cpu().numpy() for value in shap_values]  # 确保所有值在CPU上
 # 确保shap_values在CPU上
    X = Train_Value.cpu().numpy()  # 同样确保输入数据在CPU上

    # Step 2: 生成柱状图显示特征重要性
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.summary_plot(shap_values_numpy, X, plot_type="bar", show=False)
    plt.title('SHAP_numpy Sorted Feature Importance')
    plt.tight_layout()
    plt.savefig("SHAP_numpy_Sorted_Feature_Importance.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Step 3: 绘制蜂巢图和顶部特征贡献图
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)

    # 生成蜂巢图
    shap.summary_plot(shap_values_numpy, X, feature_names=X.columns, plot_type="dot", show=False, color_bar=True)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整位置以便显示热度条

    # 共享 y 轴
    ax1 = plt.gca()

    # 创建共享 y 轴的另一个图，绘制顶部的特征贡献图
    ax2 = ax1.twiny()
    shap.summary_plot(shap_values_numpy, X, plot_type="bar", show=False)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整位置以对齐

    # 添加顶部横线以提高可读性
    ax2.axhline(y=13, color='gray', linestyle='-', linewidth=1)

    # 调整透明度
    bars = ax2.patches
    for bar in bars:
        bar.set_alpha(0.2)

    # 设置标签
    ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
    ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)

    # 移动顶部的 X 轴，避免重叠
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()

    # 设置y轴标签
    ax1.set_ylabel('Features', fontsize=12)

    plt.tight_layout()
    plt.savefig("SHAP_combined_with_top_line_corrected.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    ####################全局可解释性与局部可解释性一起放在了一张图上####################




    ##输出Bar图
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, scaled_data,
                      plot_type="bar",
                      show_values_in_legend=True,
                      show=False)
    plt.savefig("figure_2.pdf")
    plt.show()

    ##输出dot图
    shap.summary_plot(shap_values[1], scaled_data,
                      plot_type="dot",
                      show_values_in_legend=True)
    plt.show()

    ##输出violin图
    scaled_data=np.array(scaled_data)
    shap.summary_plot(shap_values[1], scaled_data,
                      plot_type="violin",
                      show_values_in_legend=True)
    plt.show()
    ##单个特征的影响
    shap.dependence_plot('Feature 12', shap_values[1], scaled_data)
    plt.show()

    scaled_data = np.array(scaled_data)

    import matplotlib
    import matplotlib as mpl
    from scipy.stats import pearsonr  # 用于计算皮尔逊相关系数
    from scipy.stats import spearmanr  # 用于计算斯皮尔曼相关系数

    # 使用 TrueType 字体渲染
    mpl.rcParams['pdf.fonttype'] = 42  # 设置字体嵌入为 TrueType (42 = 使用TrueType)
    matplotlib.use('Agg')
    # 创建一个 3x5 的网格布局，大小可以根据需要调整
    # 遍历每个特征，将每个 SHAP 依赖性图放在相应的子图位置
    scaled_data = np.array(scaled_data)

    # 创建一个 3x5 的网格布局
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))  # 3行5列的图，每个子图的大小为 (4, 4)
    matplotlib.use('Agg')

    # 遍历每个特征，将每个 SHAP 依赖性图放在相应的子图位置
    for i in range(15):
        row = i // 5  # 计算第几行
        col = i % 5  # 计算第几列
        ax = axes[row, col]  # 选择相应的子图

        # 计算特征i和它的SHAP值之间的斯皮尔曼相关系数和显著性P值
        feature_values = scaled_data[:, i]
        shap_values_feature = shap_values[1][:, i]
        spearman_corr, p_value = spearmanr(feature_values, shap_values_feature)

        # 绘制依赖性图并在标题中添加相关系数和P值信息
        ax.set_title(f'Feature {i} (Spearman r = {spearman_corr:.2f}, p = {p_value:.3f})')

        # 在子图中绘制 SHAP 依赖性图
        shap.dependence_plot(i, shap_values[1], scaled_data, interaction_index=None, ax=ax)

    # 调整子图间距，防止标签重叠
    plt.tight_layout()

    # 保存整个3x5的网格图为PDF文件
    plt.savefig("xx.pdf", bbox_inches='tight')

    # 显示图像（如果需要在交互环境中展示）
    plt.show()
    shap.force_plot(explainer.expected_value,
                    shap_values[1],
                    scaled_data)
    plt.show()


    ####患者为1000395
    shap_values_single = shap.Explanation(
        values=shap_values[1][1],  # 选择第一个样本的 SHAP 值
        base_values=explainer.expected_value[0],  # 选择第一个样本的基线值
        data=scaled_data[0],  # 选择第一个样本的输入特征值
        feature_names=datasets_discovery.columns[1:16]  # 特征名称
    )
    shap.plots.waterfall(shap_values_single)


    ####患者为707008
    plt.figure()
    shap_values_single = shap.Explanation(
        values=shap_values[1][13],  # 选择第一个样本的 SHAP 值
        base_values=explainer.expected_value[0],  # 选择第一个样本的基线值
        data=scaled_data[0],  # 选择第一个样本的输入特征值
        feature_names=datasets_discovery.columns[1:16]  # 特征名称
    )
    shap.plots.waterfall(shap_values_single)
    shap.plots.heatmap()
    plt.tight_layout()  # 确保布局没有被裁剪
    plt.savefig("figure_3.pdf", bbox_inches='tight')  # 保存为 PDF 文件
    plt.show()

    ##热图可解释性
    explainer = shap.Explanation(
        values=shap_values[1],
        base_values=explainer.expected_value[0],
        data=scaled_data,  # 生成 50 个样本的特征值
        feature_names=datasets_discovery.columns[1:16]
    )
    # 使用 shap.plots.heatmap 生成 heatmap 图
    shap.plots.heatmap(explainer)
    plt.show()


