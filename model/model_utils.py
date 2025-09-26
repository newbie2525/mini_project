# model_utils.py
import pickle

def save_model(similarity_matrix, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(similarity_matrix, f)

def load_model(file_name):
    with open(file_name, 'rb') as f:
        similarity_matrix = pickle.load(f)
    return similarity_matrix
