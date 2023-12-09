import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

def ids_encoder(ratings):

    users = sorted(ratings['userId'].unique())
    movies = sorted(ratings['movieId'].unique())
    uencoder = LabelEncoder()
    iencoder = LabelEncoder()
    uencoder.fit(users)
    iencoder.fit(movies)
    ratings.userId = uencoder.transform(ratings.userId.tolist())
    ratings.movieId = iencoder.transform(ratings.movieId.tolist())
    return ratings, uencoder, iencoder

def create_model(rating_matrix, metric):
    
    model = NearestNeighbors(metric=metric, n_neighbors=26, algorithm='brute')
    model.fit(rating_matrix)    
    return model

def nn(rating_matrix, model):
    
    similarities, neighbors = model.kneighbors(rating_matrix)        
    return similarities[:, 1:], neighbors[:, 1:]

def close_movies(userId, neighbors, ratings):
    
    user_neighbors = neighbors[userId-1]
    rows = ratings.loc[ratings.userId.isin(user_neighbors)]
    freq = rows.groupby('movieId')['rating'].count().reset_index(name='count')
    freq = freq.sort_values(['count'],ascending=False)
    freq_movies = freq.movieId
    rated_movies = ratings.loc[ratings.userId == userId].movieId.to_list()
    movies = np.setdiff1d(freq_movies, rated_movies, assume_unique=True)[:20]
    return movies

def predict(userId, movieId, means, np_mean_ratings, encoded_ratings):

    model = create_model(encoded_ratings, 'cosine')
    similarities, neighbors = nn(encoded_ratings, model)
    user_similarities = similarities[userId-1]
    user_neighbors = neighbors[userId-1]
    user_mean = means[userId-1]
    user_movies = np_mean_ratings[np_mean_ratings[:, 1].astype('int') == movieId]
    similar_users = user_movies[np.isin(user_movies[:, 0], user_neighbors)]
    norm_ratings = similar_users[:,4]
    ind = [np.where(user_neighbors == uid)[0][0] for uid in similar_users[:, 0].astype('int')]
    sims = user_similarities[ind]
    numr = np.dot(norm_ratings, sims)
    denr = np.sum(np.abs(sims))
    if not numr or not denr:
        return user_mean
    r_hat = user_mean + np.dot(norm_ratings, sims) / np.sum(np.abs(sims))
    return r_hat

def user2userPredictions(userId, means, np_mean_ratings, encoded_ratings, neighbors, ratings):
    
    movies = close_movies(userId, neighbors, ratings)
    recommendations = pd.DataFrame(columns=['userId', 'movieId', 'predictedRating'])
    for movieId in movies:
        r_hat = predict(userId, movieId, means, np_mean_ratings, encoded_ratings)
        recommendations.loc[len(recommendations.index)] = [int(userId), int(movieId), round(r_hat,2)]
    return recommendations

def recommend(n, ratings):
    ratings, _, _ = ids_encoder(ratings)     
    encoded_ratings = csr_matrix(pd.crosstab(ratings.userId, ratings.movieId, ratings.rating, aggfunc=sum).fillna(0).values)
    model = create_model(encoded_ratings, 'cosine')
    _, neighbors = nn(encoded_ratings, model)

    means = ratings.groupby(by='userId', as_index=False)['rating'].mean()
    mean_ratings = pd.merge(ratings, means, suffixes=('','_mean'), on='userId')
    means = means.to_numpy()[:, 1]
    mean_ratings['norm_rating'] = mean_ratings['rating'] - mean_ratings['rating_mean']
    np_mean_ratings = mean_ratings.to_numpy()

    df = user2userPredictions(n, means, np_mean_ratings, encoded_ratings, neighbors, ratings)
    df['userId'] = df['userId'].astype('int')
    df['movieId'] = df['movieId'].astype('int')
    return df