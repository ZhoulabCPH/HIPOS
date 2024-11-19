# Load required libraries
library(dendextend)
library(pheatmap)
library(NbClust)
library(cluster)
library(factoextra)
library(mclust)
library(circlize)
library(corrplot)
library(ggcorrplot)
library(ggplot2)

# Define functions for modularity
initialize_environment <- function(path) {
  setwd(path)
  message("Working directory set to: ", path)
}

load_correlation_matrix <- function(file_path) {
  correlation_matrix <- read.csv(file_path, header = TRUE, row.names = 1)
  message("Correlation matrix loaded successfully.")
  return(correlation_matrix)
}

perform_model_based_clustering <- function(data_matrix, cluster_range = 1:40) {
  message("Performing model-based clustering using Mclust...")
  m_clust <- Mclust(as.matrix(data_matrix), G = cluster_range)
  optimal_clusters <- m_clust$G
  message("Optimal number of clusters determined: ", optimal_clusters)
  cluster_result <- Mclust(data_matrix, G = optimal_clusters)
  return(list(model = m_clust, result = cluster_result))
}

sort_correlation_matrix <- function(data_matrix, cluster_labels) {
  sorted_data <- data_matrix[order(cluster_labels), order(cluster_labels)]
  return(sorted_data)
}

generate_clustering_plots <- function(cluster_result, data_matrix, sorted_data) {
  message("Generating plots...")

  # Heatmap of the original correlation matrix
  pheatmap(data_matrix, main = "Original Correlation Matrix",
           clustering_method = "complete", border_color = NA)

  # Heatmap of the sorted correlation matrix
  pheatmap(sorted_data, main = "Sorted Correlation Matrix",
           clustering_method = "complete", border_color = NA)

  # Visualize BIC for optimal cluster selection
  plot(cluster_result$model, what = "BIC", main = "BIC for Model Selection")

  # Visualize the clustering dendrogram
  dendrogram <- as.dendrogram(cluster_result$model$classification)
  dendrogram <- color_branches(dendrogram, k = cluster_result$model$G)
  plot(dendrogram, main = "Cluster Dendrogram", xlab = "Samples", ylab = "Height")

  # Plot cluster assignments
  fviz_cluster(list(data = data_matrix, cluster = cluster_result$result$classification),
               geom = "point", main = "Cluster Visualization", repel = TRUE)

  message("Plots generated.")
}

# Main script
Cluster_HPF_Path <- "path"  # Replace with the actual path
correlation_matrix_file <- "Corrletation_matrix.csv"

initialize_environment(Cluster_HPF_Path)

correlation_matrix <- load_correlation_matrix(correlation_matrix_file)

# Perform clustering
clustering_results <- perform_model_based_clustering(correlation_matrix)
m_clust <- clustering_results$model
cluster_result <- clustering_results$result

# Sort correlation matrix by cluster classifications
sorted_data <- sort_correlation_matrix(correlation_matrix, cluster_result$classification)

# Display clustering results
message("Cluster assignments: ", toString(cluster_result$classification))
message("BIC values: ", toString(cluster_result$BIC))

# Generate plots for clustering analysis
generate_clustering_plots(clustering_results, correlation_matrix, sorted_data)

# Save outputs
write.csv(cluster_result$classification, file = "Cluster_Labels.csv", row.names = TRUE)
write.csv(sorted_data, file = "Sorted_Correlation_Matrix.csv", row.names = TRUE)
message("Cluster assignments and sorted matrix saved to files.")
