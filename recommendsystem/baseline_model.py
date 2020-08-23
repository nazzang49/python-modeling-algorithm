# https://datascienceschool.net/view-notebook/fcd3550f11ac4537acec8d18136f2066/

import surprise
import pandas as pd
import matplotlib.pyplot as plt
from surprise.model_selection import KFold
import numpy as np
from surprise.model_selection import cross_validate

# 1
def do_baseline_model_test():
    data = surprise.Dataset.load_builtin('ml-100k')

    # call raw data
    # rate -> on item by user
    df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rate", "id"])
    del df["id"]
    print(df.head(10))

    df_table = df.set_index(["user", "item"]).unstack()
    print(df_table.shape)

    # NaN -> ""
    print(df_table.iloc[212:222, 808:817].fillna(""))

    # check by graph
    plt.imshow(df_table)
    plt.grid(False)
    plt.xlabel("item")
    plt.ylabel("user")
    plt.title("rate distribution")
    plt.show()

    # optimization algorithm -> SGD / ALS
    bsl_options = {
        'method': 'als',
        'n_epochs': 5,
        'reg_u': 12,
        'reg_i': 5
    }
    algo = surprise.BaselineOnly(bsl_options)

    np.random.seed(0)
    acc = np.zeros(3)
    cv = KFold(3)
    for i, (trainset, testset) in enumerate(cv.split(data)):
        algo.fit(trainset)
        predictions = algo.test(testset)

        # evaluate functional point -> RMSE / MAE / FCP
        acc[i] = surprise.accuracy.rmse(predictions, verbose=True)

    # mean value of accuracy -> acc0 + acc1 + acc2 / 3
    print(acc.mean())

    # reduce lines in short
    # cross_validate(algo, data)


