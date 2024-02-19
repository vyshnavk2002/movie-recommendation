# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:42:56 2024

@author: user2
"""

import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import streamlit as st

with open('C:\\Users\\user2\\Downloads\\Movie_recom.sav', 'rb') as file:
    loaded_model = pickle.load(file)

movies_data = pd.read_csv('C:\\Users\\user2\\Downloads\\movies.csv')    

def movie_recommendation(movie_name):
    # Remove the redundant input statement
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        
        combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
        combined_features = combined_features.fillna('')

        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)
        similarity = cosine_similarity(feature_vectors)
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        recommended_movies = []
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            recommended_movies.append(title_from_index)

        return recommended_movies[:10]  # Display the top 10 recommended movies
    else:
        return ["No close matches found"]

def main():
    st.title('Movie Recommendation System')
    st.image('C:\\Users\\user2\\Downloads\\photo-1626814026160-2237a95fc5a0.avif', use_column_width=True)
    
    Enter_Your_Favourite_Movie_Name = st.text_input('Enter your favourite movie name:')
    
    movies = ''
    if st.button('Movie Result'):
        movies = movie_recommendation(Enter_Your_Favourite_Movie_Name)
        
    st.success(', '.join(movies))  # Join the recommended movies into a string for display

if __name__ == '__main__':
    main()
