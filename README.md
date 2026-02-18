# netflix-content-clustering

Project Overview

Netflix offers thousands of titles worldwide, but discovering content that matches user preferences can be challenging. This project applies unsupervised machine learning to group Netflix movies and TV shows into meaningful clusters based on descriptive attributes, enabling better content discovery and recommendations.

Objectives

Cluster Netflix titles using descriptive features (genres, cast, director, synopsis) without relying on user interaction data.
Identify hidden patterns in the dataset to enhance content exploration.
Support a content-based recommendation system for suggesting similar titles.

Dataset
Source: Netflix dataset (NETFLIX MOVIES AND TV SHOWS CLUSTERING (1) (1) (2) (2).csv)


Project Workflow

The project is implemented in a single script: src/netflix_project.py.

Key steps include:

1.  Import Libraries & Load Dataset – Required packages: pandas, numpy, sklearn, matplotlib, seaborn, wordcloud, etc.
2.  Basic Data Inspection – Explore dataset structure, missing values, and descriptive statistics.
3.  Data Cleaning – Handle missing values, normalize text, remove unnecessary columns.
4.  Feature Engineering – Combine and preprocess text fields (genres, cast, director, description).
5.  TF-IDF Vectorization – Convert text features into numerical vectors for clustering.
6.  Dimensionality Reduction – Apply PCA/SVD to reduce high-dimensional vectors.
7.  Determine Optimal Clusters – Use the elbow method and silhouette analysis to find the optimal number of clusters.
8.  Clustering Algorithms – Implement:
   . K-Means Clustering
   . Hierarchical Agglomerative Clustering
9.  Cluster Evaluation – Use Silhouette Score, Calinski-Harabasz Score, and Davies-Bouldin Score.
10. Visualization –
   . 2D scatter plots (SVD/PCA reduced vectors)
   . Word clouds per cluster
   . Bar charts for cluster distributions
11. Content-Based Recommendation System – Retrieve similar titles using cosine similarity between TF-IDF vectors.
