
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image
import base64

# Function to encode an image to base64
def get_base64_of_bin_file(bin_file):
    """
    Reads a binary file and returns its base64 encoded string.

    Args:
        bin_file (str): Path to the binary file.

    Returns:
        str: Base64 encoded string of the file content.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to apply background images to the main and sidebar
def apply_bg(main_bg, side_bg):
    """
    Applies background images to the main app and the sidebar in Streamlit.

    Args:
        main_bg (str): Path to the main background image file.
        side_bg (str): Path to the sidebar background image file.
    """
    main_bg_base64 = get_base64_of_bin_file(main_bg)
    side_bg_base64 = get_base64_of_bin_file(side_bg)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{main_bg_base64});
            background-size: cover;
        }}
        .sidebar .sidebar-content {{
            background: url(data:image/png;base64,{side_bg_base64});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background images
main_bg = "images/img.avif"
side_bg = "images/img.avif"
apply_bg(main_bg, side_bg)

# Load data
df = pd.read_csv("data.csv")

# Clean the artists column by removing unwanted characters
df["artists"] = df["artists"].str.replace("[", "").str.replace("]", "").str.replace("'", "")

# Create the song URL column
df["song"] = 'https://open.spotify.com/track/' + df['id'].astype(str)

# Convert song URL into clickable link
def convert(row):
    """
    Converts a row containing song information into an HTML link.

    Args:
        row (pd.Series): A row of the dataframe.

    Returns:
        str: HTML link to the song.
    """
    return '<a href="{}" target="_blank">{}</a>'.format(row['song'], row['name'])

df['song'] = df.apply(convert, axis=1)

# Normalize the feature columns for content-based filtering
scaler = StandardScaler()
feature_columns = ['valence', 'acousticness', 'danceability', 'energy', 
                   'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Content-Based Filtering Model
content_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
content_knn.fit(df[feature_columns].values)

# Collaborative Filtering Model using artists
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['artists'])

# SpotifyRecommender class with hybrid recommendation system
class SpotifyRecommender:
    """
    A music recommender system using hybrid recommendation methods.

    Attributes:
        rec_data_ (pd.DataFrame): Dataframe containing song data.
        scaler (StandardScaler): Scaler used for normalizing features.
        content_knn (NearestNeighbors): KNN model for content-based recommendations.
        tfidf (TfidfVectorizer): TF-IDF vectorizer for collaborative recommendations.
        tfidf_matrix (sparse matrix): TF-IDF matrix of the artists.
    """
    def __init__(self, rec_data):
        """
        Initializes the SpotifyRecommender with the given data.

        Args:
            rec_data (pd.DataFrame): Dataframe containing song data.
        """
        self.rec_data_ = rec_data
        self.scaler = scaler
        self.content_knn = content_knn
        self.tfidf = tfidf
        self.tfidf_matrix = tfidf_matrix

    def change_data(self, rec_data):
        """
        Updates the recommendation data.

        Args:
            rec_data (pd.DataFrame): New dataframe containing song data.
        """
        self.rec_data_ = rec_data

    def content_based_recommendations(self, song_name, num_recommendations=5):
        """
        Provides content-based recommendations for a given song.

        Args:
            song_name (str): Name of the song to get recommendations for.
            num_recommendations (int): Number of recommendations to provide.

        Returns:
            pd.DataFrame: Dataframe containing recommended songs.
        """
        song = self.rec_data_[(self.rec_data_.name.str.lower() == song_name.lower())].head(1)
        if song.empty:
            return pd.DataFrame(columns=['artists', 'song'])

        song_features = song[feature_columns].values.reshape(1, -1)
        
        distances, indices = self.content_knn.kneighbors(song_features, n_neighbors=num_recommendations + 1)
        recommended_song_indices = indices.flatten()[1:]
        
        return self.rec_data_.iloc[recommended_song_indices]

    def collaborative_recommendations(self, song_name, num_recommendations=5):
        """
        Provides collaborative recommendations for a given song based on artist similarity.

        Args:
            song_name (str): Name of the song to get recommendations for.
            num_recommendations (int): Number of recommendations to provide.

        Returns:
            pd.DataFrame: Dataframe containing recommended songs.
        """
        idx = self.rec_data_[(self.rec_data_.name.str.lower() == song_name.lower())].index
        if len(idx) == 0:
            return pd.DataFrame(columns=['artists', 'song'])

        idx = idx[0]
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim.flatten()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        
        song_indices = [i[0] for i in sim_scores]
        return self.rec_data_.iloc[song_indices]
    
    def hybrid_recommendations(self, song_name, num_recommendations=5):
        """
        Provides hybrid recommendations for a given song combining content-based and collaborative methods.

        Args:
            song_name (str): Name of the song to get recommendations for.
            num_recommendations (int): Number of recommendations to provide.

        Returns:
            pd.DataFrame: Dataframe containing recommended songs.
        """
        content_recs = self.content_based_recommendations(song_name, num_recommendations)
        collab_recs = self.collaborative_recommendations(song_name, num_recommendations)
        
        combined_recs = pd.concat([content_recs, collab_recs]).drop_duplicates().head(num_recommendations)
        return combined_recs[['artists', 'song']]

# Initialize the recommender
recommender = SpotifyRecommender(df)

# Function to get recommendations
def get_recommendations(song_name, num_recommendations):
    """
    Gets hybrid recommendations for a given song.

    Args:
        song_name (str): Name of the song to get recommendations for.
        num_recommendations (int): Number of recommendations to provide.

    Returns:
        pd.DataFrame: Dataframe containing recommended songs.
    """
    return recommender.hybrid_recommendations(song_name, num_recommendations)

# Streamlit app setup
st.title('Music Recommendation System')
st.subheader('Song Name:')
song_name = st.text_input('Enter song name:')

# Slider for number of recommendations
st.subheader("Number of Recommendations:")
no_of_r = st.slider("Select number:", 1, 10, label_visibility="collapsed")
st.subheader('Selected: {}'.format(no_of_r))

submit = st.button('Recommend')

# Display recommendations
if submit:
    recommendations = get_recommendations(song_name, no_of_r)
    if not recommendations.empty:
        st.write(recommendations.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.write("No recommendations found for the given song name.")

# Sidebar content
st.sidebar.title('Observations')
st.sidebar.write('* It will take 1.2 years for someone to listen to all the songs.')
st.sidebar.write('* An artist creating a high energy song with either electric instruments or electronic songs has the best chance of getting popular.')
st.sidebar.write('* The most popular artist from 1921–2020 is [*The Beatles*](https://open.spotify.com/artist/3WrFJ7ztbogyGnTHbHJFl2)')
st.sidebar.title('Visualization')

# Display images in the sidebar
st.sidebar.header('Most Popular Tracks')
image1 = Image.open('images/popular_tracks.png')
st.sidebar.image(image1)

st.sidebar.header('No of Tracks Added')
image2 = Image.open('images/track_added.png')
st.sidebar.image(image2)

st.sidebar.header('Correlation Map')
image4 = Image.open('images/corr.png')
st.sidebar.image(image4)

st.sidebar.header('Audio Characteristics')
image3 = Image.open('images/audioc.png')
st.sidebar.image(image3)



