import os
os.chdir('/Users/nikhil/Documents/Sem3/CS578/Project')
import pandas as pd
from sklearn.metrics import mean_squared_error
from mem_based import recommend
import sys

def _progress(num, users):
        sys.stdout.write('\rRating predictions. Progress status : %.1f%%' % (float(num/len(users))*100.0))
        sys.stdout.flush()

ratings = pd.read_csv('./dataset/ml-latest-small/ratings.csv').iloc[:,:3]
movies = pd.read_csv('./dataset/ml-latest-small/movies.csv').iloc[:,:2]

tot_loss, num = 0, 0
users = list(set(list(ratings['userId'])))

for n in users:

    n_5_ratings = ratings[ratings['userId'] == n].iloc[0:5].reset_index(drop=True)
    n_rest_ratings = ratings[ratings['userId'] == n].iloc[5:].reset_index(drop=True)
    not_n_ratings = ratings[ratings['userId'] != n].reset_index(drop=True)
    input_ratings = pd.concat([n_5_ratings, not_n_ratings], ignore_index=True)
    df = recommend(n, input_ratings)

    comparison_df = pd.merge(n_rest_ratings, df,  how='inner', on=['userId','movieId'])

    if not comparison_df.empty:
        true_ratings = list(comparison_df['rating'])
        predicted_ratings = list(comparison_df['predictedRating'])
        tot_loss += mean_squared_error(true_ratings, predicted_ratings) * len(true_ratings)

    num += 1
    _progress(num, users)

print('\nTotal Loss: ', round(tot_loss,2))
print('Loss per user: ', round(tot_loss/len(users),2))

# Total Loss:  443.96
# Loss / total users:  0.73