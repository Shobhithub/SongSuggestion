import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("SpotifyFeatures.csv")
audio_features = df[['danceability', 'energy', 'loudness', 'speechiness',
                     'instrumentalness', 'liveness', 'valence', 'tempo']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(audio_features)

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.title("Song Recommender Based on Audio Cluster")
selected_cluster = st.selectbox("select a cluster (0-3):", df['Cluster'].unique())

n = st.slider("Number of Songs", min_value=1, max_value=10, value=5)

suggested = df[df['Cluster'] == selected_cluster][['track_name', 'artist_name', 'genre']].sample(n)
st.write(" Suggested Songs:")
st.dataframe(suggested)
