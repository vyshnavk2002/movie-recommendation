# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:48:43 2024

@author: user2
"""

import pickle
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Open the file in binary mode 'rb'
with open('C:\\Users\\user2\\Downloads\\Movie_recom.sav', 'rb') as file:
    loaded_model = pickle.load(file)
movies_data = pd.read_csv('C:\\Users\\user2\\Downloads\\movies.csv')
movie_name=input('Enter your favourite movie name:')
list_of_all_titles=movies_data['title'].tolist()
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
close_match=find_close_match[0]
index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
combined_features=movies_data['genres']+''+movies_data['keywords']+''+movies_data['tagline']+''+movies_data['cast']+''+movies_data['director']
vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)
similarity=cosine_similarity(feature_vectors)
similarity_score=list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)

i=1

for movie in sorted_similar_movies:
  index=movie[0]
  title_from_index=movies_data[movies_data.index==index]['title'].values[0]
  if(i<30):
    print(i,'.',title_from_index)
    i+=1