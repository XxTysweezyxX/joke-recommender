# Shared evaluation settings for fair comparison across models

LIKE_THRESHOLD = 7.0 # minimum rating for a joke to count as liked/relevant

HOLDOUT_PER_USER = 2 # number of liked jokes held out per user for testing

K = 5  # number of top recommendations to evaluate for each user

EVAL_USERS = 500 # maximum number of users to include in evaluation

SEED = 42 # fixed random seed so results stay reproducible across runs