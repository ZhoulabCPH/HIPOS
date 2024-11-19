import pandas as pd
import numpy as np
import torch
import torch.optim
import torch.cuda.amp as amp
import argparse
import warnings
from sklearn.preprocessing import StandardScaler
from dataset import *
from models import model
from utils import yaml_config_hook, save_model

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize standard scaler
std = StandardScaler()

############################################################
# Utility Functions
############################################################
def preprocess_clinical_names(clinical_data):
    """
    Preprocess clinical names to extract meaningful information.
    """
    clinical_names = clinical_data["编码"].values
    processed_names = [name.split('-')[1] + '-' + name.split('-')[2] for name in clinical_names]
    return np.array(processed_names)

def get_predictions(values, datasets, net):
    """
    Perform inference using the model and return predictions.
    """
    test_batch = {
        'image': values,
        'name': datasets.values[:, 0]
    }
    net.eval()
    net.output_type = ['inference']
    
    with torch.no_grad():
        output = net(test_batch)
    
    predicted_labels = output['Predict'].cpu().detach().numpy()
    prediction_scores = output['Predict_score'].cpu().detach().numpy()[:, 1]

    datasets['Label'] = predicted_labels
    datasets['predict_Score'] = prediction_scores
    return datasets

############################################################
# Main Execution
############################################################
if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    args = parser.parse_args()

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.Net(arg=args).to(device)

    if args.initial_checkpoint != 'None':
        print(f"Loading checkpoint: {args.initial_checkpoint}")
        checkpoint = torch.load(args.initial_checkpoint, map_location=device)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
    else:
        start_epoch = 0

    # External dataset inference
    print("Loading external dataset...")
    external_dataset = pd.read_csv(args.External)
    external_values = torch.from_numpy(np.array(external_dataset.values[:, 1:16], dtype=np.float32)).to(device)

    print("Performing inference...")
    predicted_external = get_predictions(external_values, external_dataset, net)

    # Save predictions
    output_path = "Log/xx.csv"
    predicted_external.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
