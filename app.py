import streamlit as st
from models.recommendation_model import MusicRecommendationModel

# Load the recommendation model
model = MusicRecommendationModel("data/spotify.csv")
model.fit()

# Streamlit UI
st.title('Music Recommendation System')
st.sidebar.title('User Input')

# Get user input
song_name = st.sidebar.text_input('Enter a song name')

# Button to trigger recommendation
if st.sidebar.button('Get Recommendations'):
    try:
        recommendations = model.recommend(song_name)
        if recommendations.empty:
            st.write("Sorry, the song was not found in the dataset. Please enter another song name.")
        else:
            st.write(recommendations)
    except IndexError:
        st.write("An error occurred. Please check your input and try again.")
