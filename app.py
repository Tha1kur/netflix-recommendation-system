"""
Netlo Movie Recommendation System - Streamlit Web App
"""

import os
import pickle
import requests
import streamlit as st
import pandas as pd

# Hide Streamlit default menu & footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# GitHub release URLs
MODEL_FILES = {
    "count_vectorizer.pkl": "https://github.com/Tha1kur/netflix-recommendation-system/releases/download/v1.0/count_vectorizer.pkl",
    "movies.pkl": "https://github.com/Tha1kur/netflix-recommendation-system/releases/download/v1.0/movies.pkl",
    "movies_full.pkl": "https://github.com/Tha1kur/netflix-recommendation-system/releases/download/v1.0/movies_full.pkl",
    "similarity.pkl": "https://github.com/Tha1kur/netflix-recommendation-system/releases/download/v1.0/similarity.pkl",
}

# Download helper
def download_file(filename, url):
    with st.spinner(f"Downloading {filename}..."):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Ensure all model files exist
def ensure_model_files():
    for filename, url in MODEL_FILES.items():
        if not os.path.exists(filename):
            download_file(filename, url)

# Load models
@st.cache_data
def load_data():
    ensure_model_files()

    movies = pd.read_pickle("movies.pkl")
    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    return movies, similarity

# Load data
movies, similarity = load_data()

# Recommendation function
def recommend(movie_title):
    try:
        movie_index = movies[movies["title"] == movie_title].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [movies.iloc[i[0]]["title"] for i in movie_list]
    except:
        return []

# Streamlit UI
st.set_page_config(
    page_title="Netlo Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netlo Movie Recommendation System")
st.markdown("Discover your next favorite movie based on content similarity!")

with st.sidebar:
    st.header("About")
    st.write("""
    Content-Based Movie Recommendation System using:
    - NLP
    - Cosine Similarity
    - Machine Learning
    """)

selected_movie = st.selectbox("Select a movie", sorted(movies["title"].tolist()))

if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)

    if recommendations:
        st.subheader("Recommended Movies")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("No recommendations found.")

st.markdown("---")
st.markdown("""
<a href="https://github.com/Tha1kur/netflix-recommendation-system" target="_blank">
    <div style="text-align:center; margin-top:20px;">
        <button style="
            padding:10px 20px;
            border:none;
            background:#000;
            color:white;
            border-radius:6px;
            cursor:pointer;
            font-size:16px;
        ">
            View Project on GitHub
        </button>
    </div>
</a>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:10px; color:gray;">
    © 2026 Netlo Movie Recommendation System | Built by Mavilon Productions  
</div>
""", unsafe_allow_html=True)