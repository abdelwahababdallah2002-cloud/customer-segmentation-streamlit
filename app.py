
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("Customer Segmentation")

df = pd.read_csv("Wholesale customers data.csv")

st.write(df.head())

X = df.drop(columns=["Channel","Region"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()

ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters)

st.pyplot(fig)
