"""
Netlo Movie Recommendation System - Model Training
Content-Based Recommendation using CountVectorizer and Cosine Similarity
"""

import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
print("Loading datasets...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on movie ID
print("Merging datasets...")
movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
movies = movies[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.columns = ['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']

# Helper function to extract names from JSON strings
def extract_names(obj):
    """
    Extract names from JSON string representation of list of dictionaries.
    Returns a string of comma-separated names.
    """
    if pd.isna(obj) or obj == '':
        return ''
    
    try:
        # Parse JSON string to list
        lst = ast.literal_eval(obj)
        
        # Extract 'name' field from each dictionary (limit to first 3 for cast)
        if isinstance(lst, list):
            names = [item.get('name', '') for item in lst if isinstance(item, dict)]
            return ' '.join(names)
        return ''
    except:
        return ''

# Helper function to extract director from crew
def extract_director(crew):
    """
    Extract director name from crew JSON string.
    """
    if pd.isna(crew) or crew == '':
        return ''
    
    try:
        crew_list = ast.literal_eval(crew)
        if isinstance(crew_list, list):
            for member in crew_list:
                if isinstance(member, dict) and member.get('job') == 'Director':
                    return member.get('name', '')
        return ''
    except:
        return ''

# Clean and transform text data
print("Processing text data...")

# Extract genres
movies['genres'] = movies['genres'].apply(extract_names)

# Extract keywords
movies['keywords'] = movies['keywords'].apply(extract_names)

# Extract cast (top 3 actors)
def extract_top_cast(cast_str):
    """Extract top 3 cast members."""
    if pd.isna(cast_str) or cast_str == '':
        return ''
    try:
        cast_list = ast.literal_eval(cast_str)
        if isinstance(cast_list, list):
            # Get first 3 cast members
            names = [item.get('name', '') for item in cast_list[:3] if isinstance(item, dict)]
            return ' '.join(names)
        return ''
    except:
        return ''

movies['cast'] = movies['cast'].apply(extract_top_cast)

# Extract director
movies['director'] = movies['crew'].apply(extract_director)

# Fill missing overviews
movies['overview'] = movies['overview'].fillna('')

# Combine all features into a single string for vectorization
print("Combining features...")
movies['tags'] = (
    movies['overview'] + ' ' +
    movies['genres'] + ' ' +
    movies['keywords'] + ' ' +
    movies['cast'] + ' ' +
    movies['director']
)

# Convert to lowercase and replace spaces (for better matching)
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# Remove duplicate movies (keep first occurrence)
movies = movies.drop_duplicates(subset=['title'], keep='first')

# Reset index
movies = movies.reset_index(drop=True)

# Initialize CountVectorizer
print("Creating similarity matrix...")
cv = CountVectorizer(max_features=5000, stop_words='english')

# Transform tags to feature vectors
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate cosine similarity matrix
similarity = cosine_similarity(vectors)

# Save preprocessed data and models
print("Saving models and data...")

# Save movies dataframe (without tags column for cleaner output)
movies_output = movies[['id', 'title', 'overview']].copy()
movies_output.to_pickle('movies.pkl')

# Save similarity matrix
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

# Save CountVectorizer
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

# Save full movies data with tags (for reference)
movies.to_pickle('movies_full.pkl')

print("Training completed successfully!")
print(f"Total movies in dataset: {len(movies)}")
print(f"Similarity matrix shape: {similarity.shape}")
