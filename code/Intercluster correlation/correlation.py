import pandas as pd
import numpy as np

def load_data(feature_file, cluster_file):
    """
    Load feature and cluster data from CSV files.
    """
    try:
        patch_feature = pd.read_csv(feature_file).iloc[:, 1:]
        patch_cluster = pd.read_csv(cluster_file).iloc[:, 1:]
        print(f"Data loaded successfully: {feature_file}, {cluster_file}")
        return patch_feature, patch_cluster
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

def process_clusters(patch_feature, patch_cluster, top_n=100):
    """
    Process clusters and calculate mean features for each cluster.

    Parameters:
        patch_feature (pd.DataFrame): Feature data for patches.
        patch_cluster (pd.DataFrame): Cluster assignments for patches.
        top_n (int): Number of top patches to consider for each cluster.
    
    Returns:
        Index_feature (np.ndarray): Mean feature matrix for clusters.
        Index_cluster (np.ndarray): Cluster labels for the matrix.
    """
    label_cluster = patch_cluster.groupby(by='Pre_Label')
    index_features = []
    index_clusters = []

    for label_type, cluster_data in label_cluster:
        cluster_label = f'HPC:{label_type}'
        index_clusters.append(cluster_label)

        # Sort data within each cluster
        sorted_df = cluster_data.sort_values(by=cluster_label, ascending=False)

        # Extract top N indices, padding if fewer than N entries exist
        index_top = list(sorted_df.index)
        while len(index_top) < top_n:
            index_top.append(index_top[0])

        # Compute mean feature values for the top N patches
        cluster_mean_features = np.mean(patch_feature.values[index_top, :-1], axis=0)
        index_features.append(cluster_mean_features)

    print("Cluster processing complete.")
    return np.array(index_features, dtype=np.float32), np.array(index_clusters)

def compute_correlation_matrix(index_features, index_clusters):
    """
    Compute the correlation matrix for cluster features.

    Parameters:
        index_features (np.ndarray): Mean feature matrix for clusters.
        index_clusters (np.ndarray): Cluster labels for the matrix.

    Returns:
        pd.DataFrame: Correlation matrix as a pandas DataFrame.
    """
    correlation_matrix = np.corrcoef(index_features, rowvar=True)
    correlation_matrix_df = pd.DataFrame(correlation_matrix, columns=index_clusters, index=index_clusters)
    print("Correlation matrix computation complete.")
    return correlation_matrix_df

def save_correlation_matrix(correlation_matrix, output_file):
    """
    Save the correlation matrix to a CSV file.

    Parameters:
        correlation_matrix (pd.DataFrame): Correlation matrix to save.
        output_file (str): Path to the output file.
    """
    correlation_matrix.to_csv(output_file, index=True)
    print(f"Correlation matrix saved to {output_file}")

def main(feature_file, cluster_file, output_file, top_n=100):
    """
    Main pipeline to process features and clusters, compute correlations, and save results.

    Parameters:
        feature_file (str): Path to the patch feature CSV file.
        cluster_file (str): Path to the patch cluster CSV file.
        output_file (str): Path to save the correlation matrix.
        top_n (int): Number of top patches to consider for each cluster.
    """
    print("Starting pipeline...")
    
    # Load data
    patch_feature, patch_cluster = load_data(feature_file, cluster_file)
    
    # Process clusters
    index_features, index_clusters = process_clusters(patch_feature, patch_cluster, top_n=top_n)
    
    # Compute correlation matrix
    correlation_matrix = compute_correlation_matrix(index_features, index_clusters)
    
    # Save correlation matrix
    save_correlation_matrix(correlation_matrix, output_file)
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    # Input and output files
    feature_file = "Patch_Feature_Train.csv"  # Path to patch feature file
    cluster_file = "Patch_Cluster_Train.csv"  # Path to patch cluster file
    output_file = "Corrletation_matrix.csv"  # Path to save correlation matrix

    # Run the main pipeline
    main(feature_file, cluster_file, output_file, top_n=100)
