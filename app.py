import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

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

# Movie Selection
selected_movie = st.selectbox(
    "Select a Movie",
    movie['Movie_Title'].values
)

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

# Button
if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie_name in recommendations:
        st.write(movie_name)