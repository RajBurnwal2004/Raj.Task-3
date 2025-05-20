%%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Interactive Customer Segmentation Dashboard")
st.markdown("""
This Streamlit-based web application allows users to adjust clustering parameters, such as the number of segments. 
The dashboard includes visualizations like charts, scatter plots, and summaries to illustrate the distribution and traits of each segment. 
It also provides clear, interpretable insights into the characteristics and business relevance of each identified customer group.
""")

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

st.sidebar.header("Segmentation Controls")
k = st.sidebar.slider("Select number of customer segments", 2, 10, 4)

kmeans = KMeans(n_clusters=k, random_state=42)
df_encoded['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_encoded['PCA1'] = X_pca[:, 0]
df_encoded['PCA2'] = X_pca[:, 1]

st.subheader("Cluster Visualization")
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
ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

st.subheader("Cluster Summary")
st.write("This section summarizes the traits of each customer segment.")
summary_df = df_encoded.groupby('Cluster').mean()
st.dataframe(summary_df)

st.subheader("Business Recommendations by Segment")

recommendations = []

for cluster_id, row in summary_df.iterrows():
    rec = f"**Cluster {cluster_id}:** "
    age = row.get("Age", 0)
    spend = row.get("TotalSpend", 0)

    if age < 35:
        rec += "Young segment. Target with online promotions and trendy products. "
    elif age < 55:
        rec += "Middle-aged customers. Recommend family bundles and mid-range products. "
    else:
        rec += "Older demographic. Consider health-related or premium product marketing. "

    if spend > 800:
        rec += "High-spending group. Offer loyalty programs and premium deals."
    elif spend > 400:
        rec += "Moderate spenders. Send occasional discounts and personalized suggestions."
    else:
        rec += "Low-spending group. Use budget-friendly offers and awareness campaigns."

    recommendations.append(rec)

for rec in recommendations:
    st.markdown(rec)
