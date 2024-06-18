import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset (assuming it's stored as a CSV file)
mcdonalds = pd.read_csv('/mnt/data/mcdonalds.csv')

# Display basic information about the dataset
print("Variable names:")
print(mcdonalds.columns.tolist())
print("\nSample size:")
print(mcdonalds.shape)

# Display the first three rows of the dataset
print("\nFirst three rows:")
print(mcdonalds.head(3))

# Extract and transform segmentation variables (convert 'Yes'/'No' to binary 1/0)
segment_vars = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 
                'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']

mcdonalds_binary = mcdonalds[segment_vars].replace({'Yes': 1, 'No': 0})

# Check average values of transformed variables
avg_values = mcdonalds_binary.mean().round(2)
print("\nAverage values of transformed segmentation variables:")
print(avg_values)

# Perform Principal Component Analysis (PCA)
pca = PCA()
pca.fit(mcdonalds_binary)

# Principal components
components = pca.components_

# Plotting the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(components[0], components[1], alpha=0.8)
plt.title('Principal Component Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Adding variable names as annotations (arrows)
for i, var in enumerate(segment_vars):
    plt.arrow(0, 0, components[0, i], components[1, i], color='r', alpha=0.5, width=0.005)
    plt.text(components[0, i]*1.15, components[1, i]*1.15, var, color='g', ha='center', va='center')

plt.grid()
plt.tight_layout()
plt.show()

# Now performing k-means clustering with 2 to 8 clusters
k_range = range(2, 9)
inertia = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1234)
    kmeans.fit(mcdonalds_binary)
    inertia.append(kmeans.inertia_)

# Plotting the scree plot (Elbow method)
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Scree Plot for K-Means Clustering')
plt.xticks(k_range)
plt.grid()
plt.tight_layout()
plt.show()
