import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
current_dir = os.getcwd()
file_name = 'spotify dataset.csv'  
file_path = os.path.join(current_dir, file_name)
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File {file_name} not found in the directory {current_dir}")
    exit()
print(df.head())
print(df.info())
df = df.dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.select_dtypes(include=np.number))
sns.pairplot(df.select_dtypes(include=np.number))
plt.show()
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
kmeans = KMeans(n_clusters=5)  
clusters = kmeans.fit_predict(scaled_features)
df['Cluster'] = clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, palette='viridis')
plt.title('Clusters of Songs')
plt.show()
output_file_path = os.path.join(current_dir, 'spotify_dataset_with_clusters.csv')
df.to_csv(output_file_path, index=False)
kmeans_model = KMeans(n_clusters=5)
kmeans_model.fit(scaled_features)
model_file_path = os.path.join(current_dir, 'kmeans_model.pkl')
joblib.dump(kmeans_model, model_file_path)

new_song_features = np.array([[0.5, 0.2, 0.1, 0.3, 0.4, 0.6, 0.7, 0.8]]) 
new_song_scaled = scaler.transform(new_song_features)
cluster_prediction = kmeans_model.predict(new_song_scaled)
print(f'The predicted cluster for the new song is: {cluster_prediction[0]}')
