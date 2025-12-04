import pandas as pd
from model.preprocess import load_and_clean_data, preprocess_data
from model.recommender import create_similarity_matrix
from model.model_utils import save_model
import pickle

# Build model locally
df = load_and_clean_data('datasets/TMDB.csv', rows=5000)
new_df = preprocess_data(df)
similarity = create_similarity_matrix(new_df)

# Save both the dataframe and similarity matrix
save_model(similarity, 'model_pickle')
new_df.to_pickle('processed_data.pkl')

print("Model built and saved successfully!")