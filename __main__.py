from recommend_system_algorithm import baseline_model
from recommend_system_algorithm import collaborative_filtering
from film_recommend_engine import collaborative_filtering_model_by_user
from film_recommend_engine import collaborative_filtering_model_by_item
import numpy as np


if __name__ == "__main__":
    # collaborative filtering test by surprise package
    # baseline_model.do_baseline_model_test()
    # collaborative_filtering.do_collaborative_filtering()
    # collaborative_filtering_model_by_user.make_recommend_engine()
    collaborative_filtering_model_by_item.make_recommend_engine()

