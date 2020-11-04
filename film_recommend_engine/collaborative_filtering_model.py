import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

def make_recommend_engine():
# supervised learning
    # http://files.grouplens.org/datasets/movielens/ml-100k.zip
    # separated by tab
    df = pd.read_csv("C:/dataset/movie/ml-100k/ml-100k/u.data", sep="\t", header=None)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    print(df.head())

    # check count by group by XXXX
    print(df.groupby(["rating"])[["user_id"]].count())      # rating value min = 1 / max = 5
    print(df.groupby(["item_id"])[["user_id"]].count())     # item count = 1682

    # =============== ratings ===============

    # desc user
    n_users = df.user_id.unique().shape[0]
    print(n_users)

    # desc item
    n_items = df.item_id.unique().shape[0]
    print(n_items)

    # make 2 x 2 zero matrix = init
    ratings = np.zeros((n_users  , n_items))
    print(ratings[:])
    print(ratings.shape)

    # insert
    for row in df.itertuples():
        # 1 / row index -> user
        # 2 / col index -> movie
        # 3 / real value -> rating
        ratings[row[1]-1, row[2]-1] = row[3]

    print(type(ratings))
    print(ratings)      # zero value = no rating on the movie by the user

    # 7 : 3 = train : test
    ratings_train, ratings_test = train_test_split(ratings, test_size=0.33, random_state=42)
    print(ratings_train.shape)
    print(ratings_test.shape)

    # make similarity matrix -> square matrix (N x N matrix)
    distances = 1 - cosine_distances(ratings_train)     # myself = 1
    print(distances)
    print(distances.shape)

    # prediction and model estimation
    # np.dot() -> multiple matrix -> 631, 631 x 631, 1682 -> 631, 1682
        # about np.dot() -> https://rfriend.tistory.com/tag/np.dot%28%29
    # np.array([np.abs(distances).sum(axis=1)]) -> sum of distances about each user (axis=1 -> row)
    user_pred = distances.dot(ratings_train) / np.array([np.abs(distances).sum(axis=1)]).T      # 1, 631 -> 631, 1
    print(distances.dot(ratings_train))
    print(distances.dot(ratings_train).shape)
    print(np.array([np.abs(distances).sum(axis=1)]))
    print(np.array([np.abs(distances).sum(axis=1)]).shape)
    print(user_pred)

    # MSE
    print(get_mse(pred=user_pred, actual=ratings_train))
    # RMSE
    print(get_rmse(pred=user_pred, actual=ratings_train))

# unsupervised learning before MSE prediction
    # pick 5 people who are similar at the most
    k = 5
    neighbors = NearestNeighbors(n_neighbors=k, metric="cosine")
    neighbors.fit(ratings_train)
    top_k_distances, top_k_users = neighbors.kneighbors(ratings_train, return_distance=True)
    print("======================== top_k_distances ========================")
    print(top_k_distances)
    print("======================== top_k_users ========================")
    print(top_k_users)

    # print("======================== top_k_distances (transpose) ========================")
    # print(top_k_distances.T)
    # print("======================== top_k_users (transpose) ========================")
    # print(top_k_users.T)

    print(top_k_distances[0].T)
    print(top_k_users[0].T)

    user_pred_k = np.zeros(ratings_train.shape)

    # loop for each row
    for i in range(ratings_train.shape[0]):
        # dot -> distance * rating
        user_pred_k[i, :] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / \
                            np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T

    print("========================== user_pred_k ==========================")
    # prediction value
    print(user_pred_k)
    print(user_pred_k.shape)

    # MSE
    print(get_mse(pred=user_pred_k, actual=ratings_train))
    # RMSE
    print(get_rmse(pred=user_pred_k, actual=ratings_train))

# so, we should consider unsupervised learning before supervised learning to get more accurate results

# get MSE value
def get_mse(pred, actual):
    # flatten -> make 1-dimension
    # nonzero -> exclude zero
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

# get RMSE value
def get_rmse(pred, actual):
    # flatten -> make 1-dimension
    # nonzero -> exclude zero
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual))
