
# Deep learning-based histomorphological image phenotyping reclassifies small cell lung cancer and improves risk stratification
## Abstract
In this study, we perform a comprehensive deep learning-based histomorphological image phenotyping of SCLC patients from multi-center cohorts. Deep representation learning analysis of tissue images constructs an atlas of histomorphological phenotypes (HIPO) and reveals two robust HIPO-based SCLC subtypes correlating with distinct survival outcomes independent of molecular subtyping and clinical features. A clinically applicable image-based histomorphological phenotyping stratification system (HIPOS) was proposed to evaluate the imaging intratumor ecosystem heterogeneity and improve risk stratification in the discovery cohort of 348 patients, and validated in three independent cohorts of 94, 67 and 109 from other medical centers. The HIPOS consistently showed independent prognostic performance in predicting overall survival and disease-free survival outcomes and contributed extra prognostic significance beyond tumor–node–metastasis stage and molecular subtypes.

!["DL-CC"](./assets/DLCC.png)

## System requirements

#### Hardware Requirements

```
Device 0 name: NVIDIA GeForce RTX 4080 Laptop GPU
```

#### OS Requirements

This package is supported for Linux and Windows. The package has been tested on the following systems:

```
Linux 3.10.0-957.el7.x86_64
Windows 11 x64
```
#### Software Prerequisites

```
Python version:3.8.19
matplotlib version: 3.7.2
pandas version: 2.0.3
numpy version: 1.23.4
pytorch version: 1.11.0
CUDA available: True
CUDA version: 11.3
cuDNN enabled: True
cuDNN version: 8200
Pillow 8.2.0
opencv-python 4.5.5.64
openslide-python 1.1.1
Scikit-learn 0.24.1
R version 4.3.0
```

### Installation guide

It is recommended to install the environment in the Linux 3.10.0-957.el7.x86_64 system.

* First install Anconda3.

* Then install CUDA 11.x and cudnn.

* Finall intall these dependent python software library.

The installation is estimated to take 1 hour, depending on the network environment.




## Predictive models training

#### H&E Tile Segmentation with Watershed
  1.Convert the SVS file to PNG format.
  2.WSIs were segmented into 224x224-pixel tiles at 5x resolution.
  3.Artifacts were filtered using Otsu's thresholding, retaining tiles with ≥60% tissue coverage.
  4.Stain normalization was performed with Reinhard's method.
```
python ./data/H&E Tile Segmentation with watershed.py 
```
#### 1. Contrastive learning-based self-supervised clustering
In the file `CC_Model.py `, we provide an example of how to extract features
from each tile, given their coordinates, using a ResNet50 pre-trained on the ImageNet dataset.
The code to train such a model is available here: https://github.com/topics/resnet50.
```
python ./code/1_deeplearning_contrastive_cluster/train.py 
```
#### 2.Inter-Cluster Correlation Analysis
This code takes two CSV files as input: `Patch_Feature.csv`, which contains the feature data of image patches, and `Patch_Cluster.csv`, which includes the clustering labels for the patches. The output is a correlation matrix (`Corrletation_matrix_pd`) that quantifies the relationships between different clusters. The purpose of this code is to group image patches based on their clustering labels, calculate the mean features for each cluster, and analyze the inter-cluster correlations to better understand the relationships and similarities between the clusters, aiding in the interpretation of clustering results.
```
python ./code/1_deeplearning_contrastive_cluster/Histomorphological_Feature.py 
```
#### 3.GMM Cluster to HIPO
  1.The purpose of the `Patient_Level.py` is to aggregate patch-level cluster information into patient-level features by grouping patches based on patient identifiers and calculating the proportion of patches assigned to each HIPO type. This enables the generation of patient-level representations that can be used for further analysis, such as patient stratification or predictive modeling.
  2.The `GMM.R` aims to identify the optimal number of clusters in the similarity matrix using a model-based clustering approach (Mclust). It then classifies the matrix into clusters and reorders it for clear visualization, facilitating the exploration of patterns and relationships between the clustered entities. This process helps in deriving meaningful groupings and patterns for downstream analyses, such as feature extraction or classification.
```
python ./code/1_deeplearning_contrastive_cluster/Histomorphological_Feature.py
R ./code/1_deeplearning_contrastive_cluster/Histomorphological_Feature.py
```
#### 4.GMM Cluster to HIPO
1.The purpose of the `Patient_Level.py` is to aggregate patch-level cluster information into patient-level features by grouping patches based on patient identifiers and calculating the proportion of patches assigned to each HIPO type. This enables the generation of patient-level representations that can be used for further analysis, such as patient stratification or predictive modeling.
2.The `GMM.R` aims to identify the optimal number of clusters in the similarity matrix using a model-based clustering approach (Mclust). It then classifies the matrix into clusters and reorders it for clear visualization, facilitating the exploration of patterns and relationships between the clustered entities. This process helps in deriving meaningful groupings and patterns for downstream analyses, such as feature extraction or classification.
```
python ./code/1_deeplearning_contrastive_cluster/Histomorphological_Feature.py
R ./code/1_deeplearning_contrastive_cluster/Histomorphological_Feature.py
```

#### 5.DeepDHP
The DeepDHP architecture featured two autoencoders:
  1.A global autoencoder to extract overall tissue characteristics.
  2.A local autoencoder with an attention mechanism to focus on critical features.
Global and local embeddings were concatenated and passed through a classifier for subtype prediction.
```
python ./code/1_deeplearning_contrastive_cluster/Histomorphological_Feature.py
R ./code/1_deeplearning_contrastive_cluster/Histomorphological_Feature.py
```

