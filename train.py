import os
import random
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from IPython.display import HTML, display

# Seed everything
SEED = 9453
random.seed(SEED)
np.random.seed(SEED)


def join_dataframe(df1, df2, columns):
    df = df1.join(df2.set_index(columns), on=columns)
    return df


def rmse(y, y_prediction):
    return np.sqrt(np.mean(np.square(y - y_prediction)))


# Function to import basic features
def load_basic_dataset_with_downsampling(number_of_slice):
    # Get sales basic features and downsampling
    df_sales_features = pd.read_pickle("./features/sales_basic_features.pkl")
    ids = np.array_split(list(df_sales_features["id"].unique()), number_of_slice)[0]
    df_sales_features = df_sales_features[df_sales_features["id"].isin(ids)].reset_index(drop=True)

    # Get calendar features
    df_calendar_features = pd.read_pickle("./features/calendar_features.pkl")
    calendar_selected_columns = df_calendar_features.columns.tolist()
    calendar_selected_columns.remove("date")
    df_features = join_dataframe(df_sales_features, df_calendar_features[calendar_selected_columns], ["d"])

    # Get price features
    df_price_features = pd.read_pickle("./features/price_features.pkl")
    df_features = join_dataframe(df_features, df_price_features, ["store_id", "item_id", "d"])

    return df_features


# Function to import lag features
def load_lag_features(df):
    # Get lag features
    df_lag_features = pd.read_pickle('./features/sales_lag_features.pkl')
    lag_selected_columns = df_lag_features.columns.tolist()
    lag_selected_columns.remove("sales")
    df_features = join_dataframe(df, df_lag_features[lag_selected_columns], ["id", "d"])

    return df_features


# Function to make fast training test
def make_fast_training(df, feature_list, number_of_train, lgb_params):

    # Set aside 28 days for validation
    train_X, train_y = df[df["d"] <= (number_of_train - 28)][feature_list], df[df["d"] <= (number_of_train - 28)][
        target_feature]
    validation_X, validation_y = df[df["d"] > (number_of_train - 28)][feature_list], \
                                 df[df["d"] > (number_of_train - 28)][target_feature]
    train_data = lgb.Dataset(train_X, label=train_y)
    validation_data = lgb.Dataset(validation_X, label=validation_y)

    # Train
    estimator = lgb.train(lgb_params, train_data, valid_sets=[train_data, validation_data], verbose_eval=500)

    return estimator


# Set global variables
NUMBER_OF_SLICE = 90
NUMBER_OF_TRAIN = 1913  # Not to include test data

df_fast_basic_training = load_basic_dataset_with_downsampling(NUMBER_OF_SLICE)
df_fast_basic_training = df_fast_basic_training[df_fast_basic_training["d"] <= NUMBER_OF_TRAIN].reset_index(drop=True)
df_fast_basic_training.info()

df_fast_lag_training = load_lag_features(df_fast_basic_training)
df_fast_lag_training.info()

model_params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'None',
    'max_bin': 127,
    'bin_construct_sample_cnt': 20000000,
    'num_leaves': 2 ** 10 - 1,
    'min_data_in_leaf': 2 ** 10 - 1,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 0.1,
    'boost_from_average': False,
    'force_row_wise': True,
}

# Gather feature list
target_feature = "sales"
remove_feature_list = ["id", "d", target_feature]
feature_list = [column for column in list(df_fast_basic_training) if column not in remove_feature_list]
print(feature_list)

# Train baseline model
baseline_model = make_fast_training(df_fast_basic_training, feature_list, NUMBER_OF_TRAIN, model_params)

# Train lag model
lag_model = make_fast_training(df_fast_lag_training, feature_list, NUMBER_OF_TRAIN, model_params)

# Get validation datasets
feature_list = [column for column in list(df_fast_lag_training) if column not in remove_feature_list]
df_validation = df_fast_lag_training[df_fast_lag_training["d"] > (NUMBER_OF_TRAIN - 28)].reset_index(drop=True)

# Make predictions and calculate RMSE base
df_validation["prediction"] = lag_model.predict(df_validation[feature_list])
base_rmse = rmse(df_validation[target_feature], df_validation["prediction"])
print(f"Base RMSE: {base_rmse}")
