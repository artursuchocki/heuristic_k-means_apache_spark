# K-Means Clustering


# ------- At first, clean Python, without Apache Spark ------- 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.txt', header=None, delimiter=r"\s+")
X = dataset.iloc[:, 0:55].values
Y = dataset.iloc[:, 55:56].values
X.shape
Y.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_normalized = sc_X.fit_transform(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
wcss_normalized = []

for i in range(10, 200):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    kmeans.fit(X_normalized)
    wcss_normalized.append(kmeans.inertia_)
    
plt.plot(range(10, 200), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

plt.plot(range(10, 200), wcss_normalized)
plt.title('The Elbow Method Normalized')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Normalized')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 50, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_normalized)

y_kmeans.shape
# Visualising the clusters
from itertools import cycle
cycol = cycle('bgrcmk')

plt.scatter(X_normalized[y_kmeans == 0, 0], X_normalized[y_kmeans == 0, 1], s = 100, c = next(cycol) , label = 'Cluster 1')
#plt.scatter(X_normalized[y_kmeans == 5, 0], X_normalized[y_kmeans == 5, 1], s = 100, c = next(cycol), label = 'Cluster 5')
plt.scatter(X_normalized[y_kmeans == 10, 0], X_normalized[y_kmeans == 10, 1], s = 100, c = next(cycol), label = 'Cluster 10')
#plt.scatter(X_normalized[y_kmeans == 15, 0], X_normalized[y_kmeans == 15, 1], s = 100, c = next(cycol), label = 'Cluster 15')
plt.scatter(X_normalized[y_kmeans == 20, 0], X_normalized[y_kmeans == 20, 1], s = 100, c = next(cycol), label = 'Cluster 20')
#plt.scatter(X_normalized[y_kmeans == 25, 0], X_normalized[y_kmeans == 25, 1], s = 100, c = next(cycol) , label = 'Cluster 25')
plt.scatter(X_normalized[y_kmeans == 30, 0], X_normalized[y_kmeans == 30, 1], s = 100, c = next(cycol) , label = 'Cluster 30')
#plt.scatter(X_normalized[y_kmeans == 35, 0], X_normalized[y_kmeans == 35, 1], s = 100, c = next(cycol) , label = 'Cluster 35')
plt.scatter(X_normalized[y_kmeans == 40, 0], X_normalized[y_kmeans == 40, 1], s = 100, c = next(cycol) , label = 'Cluster 40')
#plt.scatter(X_normalized[y_kmeans == 45, 0], X_normalized[y_kmeans == 45, 1], s = 100, c = next(cycol) , label = 'Cluster 45')
plt.scatter(X_normalized[y_kmeans == 49, 0], X_normalized[y_kmeans == 49, 1], s = 100, c = next(cycol) , label = 'Cluster 50')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'yellow', label = 'Centroids')
plt.title('Clusters of normalized data')
#plt.legend()
plt.show()