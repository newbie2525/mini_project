#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def load_and_clean_data(file_path, rows=None):
    data=pd.read_csv('datasets/TMDB.csv')
    df=data[['id','title','overview','genres','keywords','poster_path']]
    df.dropna(inplace=True)
    df=df.drop_duplicates()
    df=df[0:rows]
    return df

def preprocess_data(df):
    df['overview'] = df['overview'].apply(lambda x: x.split())
    
    
    def transform(df, columns):
        for column in columns:
            df[column] = df[column].apply(lambda x: str(x).split(','))
            df[column]=df[column].apply(lambda x:[i.replace(" ","") for i in x])
        return df

    df = transform(df, ['genres', 'keywords'])
    df['tags'] = df['overview'] + df['genres'] + df['keywords']
    new_df = df.drop(['overview', 'keywords'], axis=1)
    new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
    return new_df

# In[ ]:




