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
##选择最优聚类数目进行层次聚类
Cluster_HPF_Path="path"
setwd(Cluster_HPF_Path)
correlation_matrix <- read.csv("Corrletation_matrix.csv", header = TRUE, row.names = 1)
m_clust <- Mclust(as.matrix(correlation_matrix), G=1:40)
m_clust$G
cluster_result <- Mclust(correlation_matrix, G = m_clust$G)
sorted_data <- correlation_matrix[order(cluster_result$classification),order(cluster_result$classification)]
aRRAT=cluster_result$classification
aRRAT
cluster_result$BIC
