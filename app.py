from flask import Flask, request, jsonify
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

API_KEY = "4625b830"  # OMDb API key

# === Load dataset ===
def load_data():
    movies = pd.read_csv("C:/Users/ADMIN/Desktop/Python/movie-suggestion/back/tmdb_5000_movies.csv")

    def parse_json_list(cell):
        if isinstance(cell, list):
            return cell
        if pd.isna(cell) or cell == "[]":
            return []
        try:
            data = ast.literal_eval(cell)
            return [d['name'] for d in data if isinstance(d, dict) and 'name' in d]
        except Exception:
            return []

    movies['genre_list'] = movies['genres'].apply(parse_json_list)
    movies['runtime'] = movies['runtime'].fillna(movies['runtime'].median())

    def categorize_length(r):
        if r < 90:
            return "short"
        elif r <= 150:
            return "medium"
        else:
            return "long"
    movies['length_cat'] = movies['runtime'].apply(categorize_length)

    movies['overview'] = movies['overview'].fillna("")
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(movies['overview'])
    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_data()

# === Mood Mapping ===
mood_to_genres = {
    "happy": ["Comedy", "Family", "Animation"],
    "sad": ["Drama", "Romance"],
    "thrilling": ["Action", "Thriller", "Adventure", "Crime"],
    "romantic": ["Romance", "Drama"],
    "scary": ["Horror", "Thriller"],
    "lonely": ["Drama", "Romance"],
    "depressed": ["Drama", "Documentary"],
    "missing loved ones": ["Romance", "Drama", "Family"],
    "nostalgic": ["Adventure", "Animation", "Family"],
    "motivated": ["Action", "Sport"],
    "curious": ["Mystery", "Sci-Fi", "Fantasy"],
    "calm": ["Drama", "Family"],
    "angry": ["Action", "Crime", "Thriller"],
    "hopeful": ["Adventure", "Drama", "Fantasy"]
}

# === OMDb helper ===
def get_poster(title):
    try:
        url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={title}"
        response = requests.get(url).json()
        if response.get("Poster") and response["Poster"] != "N/A":
            return response["Poster"]
    except:
        pass
    return "https://via.placeholder.com/200x300?text=No+Image"

# === Recommendation Logic ===
def recommend_smart(mood, language, length, fav_movie=None, top_n=5):
    filtered = movies.copy()

    if language:
        filtered = filtered[filtered['original_language'].str.lower() == language.lower()]
    if mood in mood_to_genres:
        genres = mood_to_genres[mood]
        filtered = filtered[filtered['genre_list'].apply(lambda g: any(genre in g for genre in genres))]
    if length:
        filtered = filtered[filtered['length_cat'] == length]

    if filtered.empty:
        return []

    if fav_movie:
        try:
            index = movies[movies['title'].str.lower() == fav_movie.lower()].index[0]
            distances = list(enumerate(similarity[index]))
            distances = sorted(distances, key=lambda x: x[1], reverse=True)
            filtered_indices = filtered.index.tolist()
            ranked = [i for i in distances if i[0] in filtered_indices][:top_n]
            results = movies.loc[[i[0] for i in ranked]]
        except IndexError:
            results = filtered.sort_values(by=['vote_average', 'popularity'], ascending=False).head(top_n)
    else:
        results = filtered.sort_values(by=['vote_average', 'popularity'], ascending=False).head(top_n)

    # Attach posters
    recommendations = []
    for _, row in results.iterrows():
        recommendations.append({
            "title": row["title"],
            "release_date": row["release_date"],
            "vote_average": row["vote_average"],
            "genres": row["genre_list"],
            "poster": get_poster(row["title"])
        })
    return recommendations

# === API Route ===
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    mood = data.get("mood")
    language = data.get("language")
    length = data.get("length")
    fav_movie = data.get("fav_movie")

    results = recommend_smart(mood, language, length, fav_movie)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
