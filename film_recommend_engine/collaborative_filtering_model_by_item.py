# 아이템 기반 영화 추천 엔진

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

def make_recommend_engine():
    df = pd.read_csv("C:/dataset/movie/ml-100k/ml-100k/u.data", sep="\t", header=None)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    print(df.head())

    # check count by group by XXXX
    print(df.groupby(["rating"])[["user_id"]].count())  # rating value min = 1 / max = 5
    print(df.groupby(["item_id"])[["user_id"]].count())  # item count = 1682

    # =============== ratings ===============

    # desc user
    n_users = df.user_id.unique().shape[0]
    print(n_users)

    # desc item
    n_items = df.item_id.unique().shape[0]
    print(n_items)

    # make 2 x 2 zero matrix = init
    ratings = np.zeros((n_users, n_items))
    print(ratings[:])
    print(ratings.shape)

    # insert
    for row in df.itertuples():
        # 1 / row index -> user
        # 2 / col index -> movie
        # 3 / real value -> rating
        ratings[row[1] - 1, row[2] - 1] = row[3]

    print(type(ratings))
    print(ratings)  # zero value = no rating on the movie by the user

    # 7 : 3 = train : test
    ratings_train, ratings_test = train_test_split(ratings, test_size=0.33, random_state=42)

# calculate cosine distance among films
    # size of films
    k = ratings_train.shape[1]
    # cosine distance among whole film rating values
    neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
    # transpose -> criteria is changed from user into film
    neigh.fit(ratings_train.T)
    item_distances, _ = neigh.kneighbors(ratings_train.T, return_distance=True)
    # 1682 x 1682 -> no.1 film : all / no.2 film : all ~
    print(item_distances.shape)
    print(item_distances)

    # ratings_train.dot(item_distances) -> 631, 1682 x 1682, 1682 -> 631 x 1682
    # np.array([np.abs(item_distances).sum(axis=1)]) -> 1 x 1682
    item_pred = ratings_train.dot(item_distances) / np.array([np.abs(item_distances).sum(axis=1)])
    print(np.array([np.abs(item_distances).sum(axis=1)]).shape)
    print(np.array([np.abs(item_distances).sum(axis=1)]))
    print(item_pred)




