# https://datascienceschool.net/view-notebook/fcd3550f11ac4537acec8d18136f2066/
# This model used in Netflix and many other companies

import surprise
import numpy as np
from surprise.model_selection import cross_validate, KFold


# matrix vector explanation
# x-axis = user
# y-axis = item
# [x, y] value = points on y-item by x-user
def do_collaborative_filtering_cross_validate(sim_options):
    data = surprise.Dataset.load_builtin('ml-100k')
    algo = surprise.KNNBasic(sim_options=sim_options)
    print(cross_validate(algo, data)["test_mae"].mean())

def do_knn_weight_calculation(sim_options):
    data = surprise.Dataset.load_builtin('ml-100k')
    algo = surprise.KNNBasic(sim_options=sim_options)
    print(cross_validate(algo, data)["test_mae"].mean())

    algo = surprise.KNNWithMeans(sim_options=sim_options)
    print(cross_validate(algo, data)["test_mae"].mean())

    algo = surprise.KNNBaseline(sim_options=sim_options)
    print(cross_validate(algo, data)["test_mae"].mean())

def do_collaborative_filtering():
    msd_list = ["msd", "cosine", "pearson", "pearson_baseline"]

    for msd in msd_list:
        sim_options = {"name": msd}
        do_collaborative_filtering_cross_validate(sim_options)
        do_knn_weight_calculation(sim_options)
