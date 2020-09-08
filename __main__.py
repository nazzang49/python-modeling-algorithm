from recommendsystem import baseline_model
from recommendsystem import collaborative_filtering
from film_recommend_engine import collaborative_filtering_model

if __name__ == "__main__":
    # collaborative filtering test by surprise package
    # baseline_model.do_baseline_model_test()
    # collaborative_filtering.do_collaborative_filtering()
    collaborative_filtering_model.make_recommend_engine()

