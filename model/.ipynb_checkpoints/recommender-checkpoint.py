#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def create_similarity_matrix(df):
    df['tags'] = df['tags'].apply(stem)
    vectors = TfidfVectorizer(min_df=3, max_features=5000, stop_words='english')
    vectors = vectors.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie, df, similarity_matrix):
    movie_index = df[df['title'] == movie].index[0]
    distances = similarity_matrix[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]
    recommendations = [df.iloc[i[0]].title for i in movies_list]
    return recommendations


# In[ ]:




