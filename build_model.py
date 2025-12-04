import pandas as pd
from model.preprocess import load_and_clean_data, preprocess_data
from model.recommender import create_similarity_matrix
from model.model_utils import save_model

print("Loading and cleaning data...")
df = load_and_clean_data('datasets/TMDB.csv', rows=1000)  # CHANGED: 5000 -> 1000

print("Preprocessing data...")
new_df = preprocess_data(df)

print("Creating similarity matrix...")
similarity = create_similarity_matrix(new_df)

print("Saving similarity matrix...")
save_model(similarity, 'model_pickle')

print("Saving processed dataframe...")
new_df.to_pickle('processed_data.pkl')

print("✓ Model built and saved successfully!")
print(f"✓ Total movies processed: {len(new_df)}")