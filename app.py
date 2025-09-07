import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# -----------------------------
# OMDb API key
# -----------------------------
OMDB_API_KEY = "cb33314a"  # Replace with your OMDb API key

# -----------------------------
# Load movies dataset
# movies.csv should have: title, genre, language, country, year (optional)
# -----------------------------
movies = pd.read_csv('movies.csv')

# -----------------------------
# Compute similarity based on title + genre
# -----------------------------
movies['combined_features'] = movies['title'].astype(str)
cv = CountVectorizer()
vectors = cv.fit_transform(movies['combined_features'])
similarity = cosine_similarity(vectors)

# -----------------------------
# Function to fetch OMDb poster, IMDb link, and plot
# -----------------------------
@lru_cache(maxsize=500)
def fetch_omdb_data(title, year=None):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
    if year:
        url += f"&y={year}"
    data = requests.get(url).json()
    try:
        data = requests.get(url).json()
        poster = data.get("Poster") if data.get("Poster") != "N/A" else "https://via.placeholder.com/300x450?text=No+Image"
        imdb_id = data.get("imdbID")
        imdb_link = f"https://www.imdb.com/title/{imdb_id}" if imdb_id else "#"
        plot = data.get("Plot") if data.get("Plot") and data.get("Plot") != "N/A" else "No description available."
        return poster, imdb_link, plot
    except:
        return "https://via.placeholder.com/300x450?text=Error", "#", "Error fetching data."

# -----------------------------
# Recommendation function
# -----------------------------
def recommend(movie_title, top_n=10):
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = list(enumerate(similarity[movie_index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    recommended_posters = []
    recommended_links = []
    recommended_plots = []

    for i in distances[1:top_n+1]:
        movie = movies.iloc[i[0]]
        poster, imdb_link, plot = fetch_omdb_data(movie['title'])
        recommended_movies.append(movie['title'])
        recommended_posters.append(poster)
        recommended_links.append(imdb_link)
        recommended_plots.append(plot)

    return recommended_movies, recommended_posters, recommended_links, recommended_plots

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("Movie Recommender System")

# Sidebar filters
st.sidebar.title("Filters")
# No filters if language/country columns are missing
filtered_movies = movies
top_n = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)

# Searchable dropdown
selected_movie = st.selectbox(
    "Search for a movie you like:",
    sorted(filtered_movies['title'].values)
)

if st.button("Show Recommendations"):
    names, posters, links, plots = recommend(selected_movie, top_n=top_n)

    # Display in rows of 5
    rows = (len(names) + 4) // 5
    for row in range(rows):
        cols = st.columns(5)
        for col, name, poster, link, plot in zip(
            cols,
            names[row*5:(row+1)*5],
            posters[row*5:(row+1)*5],
            links[row*5:(row+1)*5],
            plots[row*5:(row+1)*5]
        ):
            # Clickable title without underline
            col.markdown(
                f'<a href="{link}" target="_blank" style="text-decoration:none; color:red;"><h4>{name}</h4></a>',
                unsafe_allow_html=True
            )
            col.image(poster, use_container_width=True)
            col.caption(plot)
