"""
Netlo Movie Recommendation System - Streamlit Web App
"""

import streamlit as st
import pickle
import pandas as pd
import os
import requests

# Model file URLs
MOVIES_URL = "https://github.com/Tha1kur/netlo-movie-recommendation-system/releases/download/v1.0/movies.pkl"
SIMILARITY_URL = "https://github.com/Tha1kur/netlo-movie-recommendation-system/releases/download/v1.0/similarity.pkl"

def download_file(url, filename):
    """Download a file from URL and save it locally."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Load preprocessed data and models
@st.cache_data
def load_data():
    """Load movies data and similarity matrix."""
    # Check and download movies.pkl if missing
    if not os.path.exists('movies.pkl'):
        with st.spinner("Downloading model files..."):
            download_file(MOVIES_URL, 'movies.pkl')
    
    # Check and download similarity.pkl if missing
    if not os.path.exists('similarity.pkl'):
        with st.spinner("Downloading model files..."):
            download_file(SIMILARITY_URL, 'similarity.pkl')
    
    try:
        movies = pd.read_pickle('movies.pkl')
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        return movies, similarity
    except FileNotFoundError:
        st.error("Model files not found. Please run model.py first to train the model.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

# Load data
movies, similarity = load_data()

# Recommendation function
def recommend(movie_title):
    """
    Recommend top 5 similar movies based on content similarity.
    
    Args:
        movie_title: Title of the movie
        
    Returns:
        List of recommended movie titles
    """
    try:
        # Find movie index
        movie_index = movies[movies['title'] == movie_title].index[0]
        
        # Get similarity scores for this movie
        distances = similarity[movie_index]
        
        # Sort movies by similarity (descending order)
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        # Extract recommended movie titles
        recommended_movies = []
        for i in movie_list:
            recommended_movies.append(movies.iloc[i[0]]['title'])
        
        return recommended_movies
    except IndexError:
        return []

# Streamlit UI
st.set_page_config(
    page_title="Netlo Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Title and header
st.title("🎬 Netlo Movie Recommendation System")
st.markdown("---")
st.markdown("### Discover your next favorite movie based on content similarity!")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This recommendation system uses:
    - **Content-Based Filtering**
    - **Natural Language Processing (NLP)**
    - **Cosine Similarity**
    
    The model analyzes movie features like:
    - Genres
    - Keywords
    - Cast
    - Director
    - Overview
    """)
    st.markdown("---")
    st.write(f"**Total Movies:** {len(movies)}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Movie selection dropdown
    st.subheader("Select a Movie")
    selected_movie = st.selectbox(
        "Choose a movie from the list:",
        options=sorted(movies['title'].tolist()),
        index=0,
        help="Select a movie to get recommendations"
    )

# Display selected movie info
if selected_movie:
    movie_info = movies[movies['title'] == selected_movie].iloc[0]
    
    with col2:
        st.subheader("Selected Movie")
        st.write(f"**Title:** {movie_info['title']}")
        if pd.notna(movie_info['overview']) and movie_info['overview'] != '':
            st.write(f"**Overview:** {movie_info['overview'][:200]}...")
    
    st.markdown("---")

# Recommendation button
st.subheader("Get Recommendations")
recommend_button = st.button("🎯 Recommend Similar Movies", type="primary", use_container_width=True)

# Display recommendations
if recommend_button:
    if selected_movie:
        with st.spinner("Finding similar movies..."):
            recommendations = recommend(selected_movie)
        
        if recommendations:
            st.success(f"Here are 5 movies similar to **{selected_movie}**:")
            st.markdown("---")
            
            # Display recommendations in a nice format
            for idx, movie in enumerate(recommendations, 1):
                movie_data = movies[movies['title'] == movie].iloc[0]
                
                with st.container():
                    col_a, col_b = st.columns([1, 4])
                    
                    with col_a:
                        st.markdown(f"### #{idx}")
                    
                    with col_b:
                        st.markdown(f"#### {movie}")
                        if pd.notna(movie_data['overview']) and movie_data['overview'] != '':
                            st.write(movie_data['overview'])
                
                if idx < len(recommendations):
                    st.markdown("---")
        else:
            st.error(f"Could not find recommendations for '{selected_movie}'. Please try another movie.")
    else:
        st.warning("Please select a movie first!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Powered by Machine Learning & NLP | Content-Based Recommendation System</p>
    </div>
    """,
    unsafe_allow_html=True
)
