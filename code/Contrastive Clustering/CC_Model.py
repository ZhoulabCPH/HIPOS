import os
import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader, RandomSampler
import tables
import warnings
from models import resnet, model
from dataset import CreatDataset, null_collate, Transform_
from utils import yaml_config_hook
import torch.cuda.amp as amp
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')


############################################################
# Utility Functions
############################################################

def get_learning_rate(optimizer):
    """
    Retrieve the current learning rate from the optimizer.
    """
    return optimizer.param_groups[0]['lr']


def initialize_metrics():
    """
    Initialize metric placeholders for validation.
    """
    return {
        "sample_names": np.array([], dtype=np.str_),
        "weights": np.array([], dtype=np.float32).reshape(-1, 1),
        "predictions": np.array([], dtype=np.float32).reshape(-1, 1),
        "true_labels": np.array([], dtype=np.float32).reshape(-1, 1),
        "instance_features": np.zeros((0, 128), dtype=np.float32),
        "cluster_features": np.zeros((0, 64), dtype=np.float32)
    }


def compute_cluster_ratios(dataframe, max_clusters=63):
    """
    Compute cluster ratios for patient-level data.
    """
    dataframe['Patient'] = dataframe['Name'].str.split('_').str[0]
    patient_data = []

    for patient, group in dataframe.groupby('Patient'):
        cluster_counts = np.zeros(max_clusters + 1, dtype=np.float32)
        for cluster in group['Pre_Label']:
            cluster_counts[int(cluster)] += 1
        cluster_ratios = cluster_counts / np.sum(cluster_counts)
        patient_data.append(np.hstack([[patient], cluster_ratios]))

    cluster_columns = ['Cluster:{}'.format(i) for i in range(max_clusters + 1)]
    result_df = pd.DataFrame(patient_data, columns=['Patient'] + cluster_columns)
    return result_df


def load_clinical_data(file_path, columns=None):
    """
    Load clinical data from a CSV file and filter necessary columns.
    """
    clinical_df = pd.read_csv(file_path)
    if columns:
        clinical_df = clinical_df[columns]
    return clinical_df


def merge_cluster_clinical(cluster_df, clinical_df):
    """
    Merge cluster data with clinical information.
    """
    return pd.merge(cluster_df, clinical_df, how='inner', on='Patient')


############################################################
# Main Training/Validation Loop
############################################################

def validate_model(net, dataloader, args, device, scaler):
    """
    Validate the model and compute metrics.
    """
    metrics = initialize_metrics()
    net.eval()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print(f"Validation progress: {batch_idx * 100 / len(dataloader):.2f}%")

        with torch.no_grad():
            batch['image_Argument1'] = batch['image_Argument1'].to(device)
            batch['image_Argument2'] = batch['image_Argument2'].to(device)
            output = net(batch)

            metrics["predictions"] = np.hstack((metrics["predictions"], output['Cluste'].cpu().numpy().reshape(1, -1)))
            metrics["cluster_features"] = np.vstack((metrics["cluster_features"], output['probability_Ci'].cpu().numpy()))
            metrics["instance_features"] = np.vstack((metrics["instance_features"], output['instance_projector_i'].cpu().numpy()))
            metrics["sample_names"] = np.hstack((metrics["sample_names"], batch['name']))

    return metrics


def save_metrics_to_csv(metrics, args, output_path):
    """
    Save model predictions and cluster features to a CSV file.
    """
    cluster_columns = ["Name"] + ["Cluster:{}".format(i) for i in range(args.Cluster_num)] + ['Pre_Label']
    cluster_features = np.hstack((
        metrics["sample_names"].reshape(-1, 1),
        metrics["cluster_features"],
        metrics["predictions"].reshape(-1, 1)
    ))
    df = pd.DataFrame(cluster_features, columns=cluster_columns)
    df.to_csv(output_path, index=False)


############################################################
# Main Function
############################################################

def main():
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # Load model and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_checkpoint = args.initial_checkpoint
    batch_size = args.batch_size

    # Load dataset
    train_df = pd.read_csv("xx.csv")
    store = tables.open_file(args.H5DTrain_Image, mode='r')
    train_patches = store.root.patches
    train_dataset = CreatDataset(train_df, train_patches, Transform_)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=null_collate
    )

    # Initialize model
    res = resnet.get_resnet("ResNet50")
    net = model.Net(args=args, resnet=res).to(device)
    scaler = amp.GradScaler(enabled=args.use_amp)

    if initial_checkpoint != 'None':
        checkpoint = torch.load(initial_checkpoint, map_location=device)
        net.load_state_dict(checkpoint['state_dict'], strict=False)

    # Validate model
    metrics = validate_model(net, train_loader, args, device, scaler)

    # Save metrics to CSV
    output_path = "Log/xx.csv"
    save_metrics_to_csv(metrics, args, output_path)


if __name__ == '__main__':
    main()
