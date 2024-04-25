import pandas as pd
from sklearn.neighbors import NearestNeighbors

class MusicRecommendationModel:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.numeric_columns = None  # Initialize as None

    def fit(self):
        # Exclude non-numeric columns from the data used for modeling
        self.numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.model.fit(self.data[self.numeric_columns])

    def recommend(self, song_name, amount=5):
        song_index = self.data[self.data['name'].str.lower() == song_name.lower()].index[0]
        distances, indices = self.model.kneighbors([self.data[self.numeric_columns].iloc[song_index]], n_neighbors=amount)
        recommended_songs = self.data.iloc[indices[0]][['name', 'artists', 'year']]
        return recommended_songs
