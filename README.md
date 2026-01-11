# Netlo Movie Recommendation System

An AI-powered movie recommendation system built using Machine Learning and Python. 
It suggests similar movies based on content similarity using NLP and cosine similarity, 
with an interactive web app built using Streamlit.

## Features

- Content-based recommendation engine
- NLP-based similarity matching
- Interactive web interface
- Real-time movie recommendations

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP
- Streamlit

## 📁 Project Structure

```
Netflix-Recommendation-System/
│
├── model.py                  # Data processing and model training
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── tmdb_5000_movies.csv      # Movies dataset
├── tmdb_5000_credits.csv     # Credits dataset
├── movies.pkl                # Preprocessed movies data (generated)
├── similarity.pkl            # Similarity matrix (generated)
├── count_vectorizer.pkl      # Trained vectorizer (generated)
└── movies_full.pkl           # Full movies data with tags (generated)
```

## Setup Instructions

1. Clone the repository
2. Download TMDB 5000 dataset from Kaggle:
   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
3. Place these files in the project folder:
   - tmdb_5000_movies.csv
   - tmdb_5000_credits.csv

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Train the model:
   ```bash
   python model.py
   ```

6. Run the web app:
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

1. **Select a Movie**: Choose a movie from the dropdown menu
2. **Click Recommend**: Press the "Recommend Similar Movies" button
3. **View Results**: See the top 5 similar movies with their overviews

## 🔧 How It Works

### Data Preprocessing

1. **Data Loading**: Loads TMDB 5000 movies and credits datasets
2. **Data Merging**: Merges movies and credits data on movie ID
3. **Feature Extraction**:
   - Extracts genre names from JSON strings
   - Extracts keywords
   - Extracts top 3 cast members
   - Extracts director name from crew
4. **Feature Combination**: Combines overview, genres, keywords, cast, and director into a single text feature

### Model Training

1. **Vectorization**: Uses CountVectorizer to convert text features to numerical vectors (max 5000 features)
2. **Similarity Calculation**: Computes cosine similarity matrix between all movie pairs
3. **Model Persistence**: Saves similarity matrix and vectorizer using pickle

### Recommendation Algorithm

1. Finds the index of the selected movie
2. Retrieves similarity scores for that movie
3. Sorts movies by similarity score (descending)
4. Returns top 5 most similar movies (excluding the selected movie itself)

## 📊 Dataset

The project uses the **TMDB 5000 Movie Dataset**:
- `tmdb_5000_movies.csv`: Contains movie metadata (genres, keywords, overview, etc.)
- `tmdb_5000_credits.csv`: Contains cast and crew information

## 🎯 Example Recommendations

When you select "Avatar", the system might recommend:
- Movies with similar sci-fi/fantasy themes
- Films with similar visual effects focus
- Movies with overlapping genres or keywords

## ⚙️ Configuration

You can modify the following parameters in `model.py`:

- `max_features`: Maximum features in CountVectorizer (default: 5000)
- Number of recommendations: Currently set to 5 (can be changed in `recommend()` function)

## 📝 Notes

- First run requires training the model (`python model.py`)
- The model training process may take a few minutes depending on your system
- All pickle files are generated during training and are required for the app to run
- The app uses Streamlit's caching to load models efficiently

## 🔄 Future Enhancements

- Collaborative filtering hybrid approach
- User ratings integration
- Movie poster display
- More sophisticated NLP techniques (TF-IDF, Word2Vec)
- Filtering by genre or year
- Movie details page with trailers

## 📄 License

This project is for educational purposes.

## 👤 Author

Netlo Movie Recommendation System

---

**Happy Movie Watching! 🍿🎬**
