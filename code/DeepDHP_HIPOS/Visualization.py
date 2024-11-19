import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score, f1_score, average_precision_score
import torch
import torch.optim
from sklearn.preprocessing import StandardScaler
from utils import yaml_config_hook
from models import attention_model
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize standard scaler
std = StandardScaler()

############################################################
# Utility Functions
############################################################
def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

def preprocess_clinical_names(clinical_data):
    """
    Preprocess clinical names from the dataset.
    """
    clinical_names = clinical_data["编码"].values
    return np.array([name.split('-')[1] + '-' + name.split('-')[2] for name in clinical_names])

def predict_model(data, datasets, model, dataset_type):
    """
    Perform inference using the model and return predictions.
    """
    model.eval()
    model.output_type = ['inference']
    with torch.no_grad():
        output = model(data)
    predicted_labels = torch.argmax(output, dim=1).cpu().detach().numpy()
    prediction_scores = output[:, 1].cpu().detach().numpy()

    if dataset_type == "Discovery":
        survival_data = datasets[['Sample_Name', 'OS', 'OSState', 'DFS', 'DFSState', 'Label']]
    else:
        survival_data = datasets[['Sample_Name', 'OS', 'OSState', 'DFS', 'DFSState']]
    
    survival_data['predict_Label'] = predicted_labels
    survival_data['predict_Score'] = prediction_scores
    return survival_data

def l1_regularization(net, lambda1):
    """
    Compute L1 regularization loss for a given model.
    """
    l1_loss = sum(torch.sum(torch.abs(param)) for param in net.parameters())
    return lambda1 * l1_loss

def compute_metrics(true_labels, predicted_scores, predicted_labels):
    """
    Compute evaluation metrics for the model's predictions.
    """
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr, tpr)

    metrics_dict = {
        "ACC": accuracy_score(true_labels, predicted_labels),
        "AUC": roc_auc_score(true_labels, predicted_scores),
        "AUPR": average_precision_score(true_labels, predicted_scores),
        "F1Score": f1_score(true_labels, predicted_labels),
        "ROC Curve": (fpr, tpr, roc_auc)
    }
    return metrics_dict

def plot_shap_summary(shap_values, data, feature_names, save_path=None):
    """
    Plot SHAP summary (bar and dot) and save if required.
    """
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, data, feature_names=feature_names, plot_type="bar", show=False)
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

    shap.summary_plot(shap_values, data, feature_names=feature_names, plot_type="dot")
    plt.show()

def shap_dependence_plots(shap_values, data, feature_names, save_path=None):
    """
    Plot SHAP dependence plots for all features.
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    for i, ax in enumerate(axes.flatten()):
        if i >= len(feature_names):
            break
        shap.dependence_plot(i, shap_values, data, feature_names=feature_names, ax=ax)
        ax.set_title(f"Feature {feature_names[i]}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

############################################################
# Main Execution
############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # Model and dataset setup
    net = attention_model.Net(arg=args).to('cpu')
    if args.initial_checkpoint != 'None':
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'], strict=False)

    train_data = pd.read_csv("xx.csv").iloc[:, 1:]
    train_data['TrueLabel'] = train_data['Label'] - 1

    # Preprocessing
    feature_values = np.array(train_data.iloc[:, 1:16], dtype=np.float32)
    train_tensor = torch.from_numpy(feature_values)

    # Inference
    pre_train = predict_model(train_tensor, train_data, net, "Discovery")
    true_labels = pre_train['Label'].values - 1
    predictions = pre_train['predict_Score'].values

    # Compute metrics
    metrics_dict = compute_metrics(true_labels, predictions, pre_train['predict_Label'])
    for metric, value in metrics_dict.items():
        if metric != "ROC Curve":
            print(f"{metric}: {value}")

    # Plot ROC curve
    fpr, tpr, roc_auc = metrics_dict["ROC Curve"]
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # SHAP explainability
    net.eval()
    explainer = shap.DeepExplainer(net, train_tensor)
    shap_values = explainer.shap_values(train_tensor)

    # Feature names
    feature_names = [f"Feature {i}" for i in range(train_tensor.shape[1])]

    # SHAP summary plots
    plot_shap_summary(shap_values, train_tensor.numpy(), feature_names, save_path="shap_summary.pdf")

    # SHAP dependence plots
    shap_dependence_plots(shap_values, train_tensor.numpy(), feature_names, save_path="shap_dependence.pdf")
