%%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Customer Segmentation App")

file_path = '/content/drive/MyDrive/Customer_Segmentation_Dataset.csv'
df = pd.read_csv(file_path, sep=',')

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df.fillna(df.mode().iloc[0], inplace=True)

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

df['Age'] = 2025 - df['Year_Birth']

df['TotalSpend'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                       'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

df_cleaned = df.drop(columns=['ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue'])

df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

k = st.slider("Select number of hehehehehe clusters", 2, 10, 4)

kmeans = KMeans(n_clusters=k, random_state=42)
df_encoded['Cluster'] = kmeans.fit_predict(X_scaled)

st.write("Clustered Data:")
st.write(df_encoded.head())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_encoded['PCA1'] = X_pca[:, 0]
df_encoded['PCA2'] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(8, 5))

unique_clusters = sorted(df_encoded['Cluster'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

for cluster_label, color in zip(unique_clusters, colors):
    cluster_data = df_encoded[df_encoded['Cluster'] == cluster_label]
    ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'],
               label=f'Cluster {cluster_label}', color=color, alpha=0.6)

ax.set_title('Customer Clusters (PCA-reduced)')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')  # Puts legend on the side
st.pyplot(fig)
