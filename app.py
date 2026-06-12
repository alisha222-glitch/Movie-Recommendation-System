import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDB API KEY
API_KEY = "a9f85b7662132b27d8b7ab4e4cc1fc05"

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

# Load Dataset
movie = pd.read_csv("Dataset for MRS.csv")

# Fill Missing Values
movie['Movie_Keywords'] = movie['Movie_Keywords'].fillna('')
movie['Movie_Overview'] = movie['Movie_Overview'].fillna('')
movie['Movie_Cast'] = movie['Movie_Cast'].fillna('')
movie['Movie_Director'] = movie['Movie_Director'].fillna('')
movie['Movie_Genre'] = movie['Movie_Genre'].fillna('')

# Create Tags
movie['tags'] = (
    movie['Movie_Genre'] + ' ' +
    movie['Movie_Keywords'] + ' ' +
    movie['Movie_Overview'] + ' ' +
    movie['Movie_Cast'] + ' ' +
    movie['Movie_Director']
)

# Convert Text to Vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movie['tags']).toarray()

# Similarity Matrix
similarity = cosine_similarity(vectors)

# Poster Function
def get_movie_poster(movie_name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
        response = requests.get(url).json()

        if response["results"]:
            poster_path = response["results"][0]["poster_path"]

            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
    except:
        pass

    return None

# Recommendation Function
def recommend(movie_name):
    movie_index = movie[movie['Movie_Title'] == movie_name].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movie.iloc[i[0]].Movie_Title for i in movie_list]

# Movie Selection
selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie['Movie_Title'].dropna().values
)

# Recommendation Button
if st.button("Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    cols = st.columns(5)

    for i, movie_name in enumerate(recommendations):

        with cols[i]:

            poster = get_movie_poster(movie_name)

            if poster:
                st.image(poster, use_container_width=True)

            st.markdown(
                f"<center><b>{movie_name}</b></center>",
                unsafe_allow_html=True
            )

            trailer_link = (
                f"https://www.youtube.com/results?search_query="
                f"{movie_name}+official+trailer"
            )

            st.markdown(
                f"<center><a href='{trailer_link}' target='_blank'>▶ Trailer</a></center>",
                unsafe_allow_html=True
            )
