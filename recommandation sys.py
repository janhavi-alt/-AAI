import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Sample Hindi movie ratings dataset
data = {
    "UserID": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    "MovieID": [101, 102, 101, 103, 102, 103, 101, 104, 103, 104],
    "Rating": [5, 4, 3, 4, 2, 5, 5, 3, 4, 2]
}

movies = {
    "MovieID": [101, 102, 103, 104],
    "Title": ["Dangal (2016)", "3 Idiots (2009)", "Chhichhore (2019)", "Lagaan (2001)"]
}

# Convert dictionaries to DataFrames
ratings = pd.DataFrame(data)
movies = pd.DataFrame(movies)

# Merge ratings with movies to get movie titles
ratings = pd.merge(ratings, movies, on="MovieID")

# Create a user-item matrix
user_item_matrix = ratings.pivot_table(index="UserID", columns="Title", values="Rating").fillna(0)

# Perform matrix factorization using Truncated SVD
svd = TruncatedSVD(n_components=2)  # Use a smaller number of components for simplicity
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_

# Reconstruct the user-item matrix
predicted_ratings = np.dot(user_factors, item_factors)

# Normalize predicted ratings to 1–5 scale
min_rating, max_rating = predicted_ratings.min(), predicted_ratings.max()
predicted_ratings = 1 + 4 * (predicted_ratings - min_rating) / (max_rating - min_rating)

# Convert predicted ratings to DataFrame
predicted_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Round predicted ratings for better readability
predicted_df = predicted_df.round()

print("Predicted Ratings (Scaled):\n", predicted_df)

# Recommend top movies for a specific user
def recommend_movies(user_id, user_item_matrix, predicted_df, top_n=3, threshold=2):
    # Find the user's predicted ratings
    user_ratings = predicted_df.loc[user_id]

    # Mask already rated movies
    rated_movies = user_item_matrix.loc[user_id] > 0
    recommendations = user_ratings[~rated_movies]

    # Debugging: Show predicted ratings for the user
    print(f"Predicted ratings for User {user_id}:\n", user_ratings)

    # Exclude low ratings and recommend high-rated movies (above threshold)
    recommendations = recommendations[recommendations > threshold].sort_values(ascending=False)

    if recommendations.empty:
        print("No recommendations above threshold. Returning top N highest-rated movies.")
        # If no recommendations meet the threshold, return the top N highest-rated movies
        recommendations = user_ratings.sort_values(ascending=False)

    return recommendations.head(top_n)

# Example recommendation for User 1
recommendations = recommend_movies(1, user_item_matrix, predicted_df, top_n=3, threshold=2)
print("\nRecommended Movies for User 1 (High Ratings):\n", recommendations)
