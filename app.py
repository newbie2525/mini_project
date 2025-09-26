from flask import Flask, request, jsonify, render_template
import pandas as pd
from model.preprocess import load_and_clean_data, preprocess_data
from model.recommender import create_similarity_matrix
from model.model_utils import save_model, load_model

app = Flask(__name__)

df = load_and_clean_data('datasets/TMDB.csv', rows=5000)
new_df = preprocess_data(df)

similarity = create_similarity_matrix(new_df)
save_model(similarity, 'model_pickle')
similarity = load_model('model_pickle')


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
            genres = ', '.join(movie_data['genres']) 
            recommended_movies.append({
                "title": movie_data.title,
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
    genres = ', '.join(movie_data['genres'])  
    return {
        "title": movie_data.title,
        "poster": poster_url,
        "genres": genres
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    movie_titles = request.form.get('movie_title').split(',')
    movie_titles = [title.strip() for title in movie_titles]  # Clean whitespace
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

if __name__ == '__main__':
    app.run(debug=True)
