from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load pre-computed model and data
MODEL_PATH = 'model_pickle.pkl'  # Adjust extension if needed
DATA_PATH = 'processed_data.pkl'

# Global variables for model and data
similarity = None
new_df = None

def load_models():
    global similarity, new_df
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Model files not found. Run build_model.py first!")
    
    # Load the pre-computed similarity matrix
    with open(MODEL_PATH, 'rb') as f:
        similarity = pickle.load(f)
    
    # Load the pre-processed dataframe
    new_df = pd.read_pickle(DATA_PATH)
    
    print(f"Loaded {len(new_df)} movies")

# Load models once when the app starts
load_models()


def get_recommendations(movie_titles):
    recommended_movies = []
    if not movie_titles:
        return recommended_movies

    all_distances = []

    for title in movie_titles:
        if title in new_df['title'].values:
            movie_index = new_df[new_df['title'] == title].index[0]
            distances = similarity[movie_index]
            all_distances.append(distances)

    if all_distances:
        # Average the distances
        avg_distances = sum(all_distances) / len(all_distances)
        movie_indices = sorted(list(enumerate(avg_distances)), reverse=True, key=lambda x: x[1])[1:10]

        for i in movie_indices:
            movie_data = new_df.iloc[i[0]]
            poster_url = f"https://image.tmdb.org/t/p/w200{movie_data['poster_path']}"
            genres = ', '.join(movie_data['genres']) if isinstance(movie_data['genres'], list) else movie_data['genres']
            recommended_movies.append({
                "title": movie_data['title'],
                "poster": poster_url,
                "genres": genres
            })

    return recommended_movies


def get_movie_details(movie_title):
    movie = new_df[new_df['title'] == movie_title]
    if movie.empty:
        return None
    movie_data = movie.iloc[0]
    poster_url = f"https://image.tmdb.org/t/p/w200{movie_data['poster_path']}"
    genres = ', '.join(movie_data['genres']) if isinstance(movie_data['genres'], list) else movie_data['genres']
    return {
        "title": movie_data['title'],
        "poster": poster_url,
        "genres": genres
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    movie_titles = request.form.get('movie_title', '').split(',')
    movie_titles = [title.strip() for title in movie_titles if title.strip()]
    
    user_movies = [get_movie_details(title) for title in movie_titles if title in new_df['title'].values]
    recommendations = get_recommendations(movie_titles)

    response = {
        "user_movies": user_movies,
        "recommendations": recommendations
    }
    return jsonify(response)

@app.route('/get_movie_titles')
def get_movie_titles():
    titles = new_df['title'].dropna().unique().tolist()  
    return jsonify(titles)

# For Vercel
app = app