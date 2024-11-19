import pandas as pd
import numpy as np

def load_data(patch_file, cluster_file):
    """
    Load patch-level data and inter-cluster correlation mapping.
    """
    try:
        patch_level = pd.read_csv(patch_file)[['Name', 'Pre_Label']]
        inter_cluster_corr = pd.read_csv(cluster_file)
        print(f"Loaded patch-level data from {patch_file}")
        print(f"Loaded inter-cluster correlation data from {cluster_file}")
        return patch_level, inter_cluster_corr
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

def map_hipo_labels(patch_level, inter_cluster_corr):
    """
    Map HIPO labels to patch-level data using a replacement dictionary.
    """
    replace_dict = dict(zip(inter_cluster_corr['Cluster'], inter_cluster_corr['HIPO']))
    patch_level['HIPO'] = patch_level['Pre_Label'].replace(replace_dict)
    patch_level['Patient'] = patch_level['Name'].apply(lambda x: x.split('_')[0])
    print("Mapped HIPO labels and extracted Patient IDs.")
    return patch_level

def construct_patient_level_features(patch_level):
    """
    Construct patient-level features based on HIPO proportions.
    """
    # Count occurrences of each HIPO for each patient
    count_df = patch_level.groupby(['Patient', 'HIPO']).size().unstack(fill_value=0)
    
    # Normalize counts to proportions
    proportion_df = count_df.div(count_df.sum(axis=1), axis=0)
    proportion_df.columns = [f'HIPO{i}' for i in proportion_df.columns]
    
    print("Constructed patient-level features.")
    return proportion_df

def save_patient_level_features(patient_level, output_file):
    """
    Save the patient-level features to a CSV file.
    """
    patient_level.to_csv(output_file, index=True)
    print(f"Saved patient-level features to {output_file}")

def main(patch_file, cluster_file, output_file):
    """
    Main function to process patch-level data and construct patient-level features.
    """
    print("Starting patient-level feature construction pipeline...")
    
    # Load data
    patch_level, inter_cluster_corr = load_data(patch_file, cluster_file)
    
    # Map HIPO labels
    patch_level = map_hipo_labels(patch_level, inter_cluster_corr)
    
    # Construct patient-level features
    patient_level = construct_patient_level_features(patch_level)
    
    # Save results
    save_patient_level_features(patient_level, output_file)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    # Input file paths
    patch_file = "xx.csv"  # Path to patch-level data
    cluster_file = "xx.csv"  # Path to inter-cluster correlation data
    output_file = "./Log/xx.csv"  # Path to save patient-level features

    # Run the main pipeline
    main(patch_file, cluster_file, output_file)
