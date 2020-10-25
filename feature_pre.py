import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, random
from math import ceil
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
from utils import *


class FeaturePre():
    def __init__(self, data_dir, features_dir):
        self.data_dir = data_dir
        self.features_dir = features_dir

        self.number_of_train = 1913
        self.days_to_predict = 28
        self.index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        self.df_sales_train_validation = pd.read_csv(self.data_dir + 'sales_train_validation.csv')

        self.calendar_df = pd.read_csv(self.data_dir + 'calendar.csv')
        self.prices_df = pd.read_csv(self.data_dir + 'sell_prices.csv')

        self.rolling_days = [7, 14, 28  ]

    def merge_by_concat(self, df1, df2, columns):
        df_temp = df1[columns]
        df_temp = df_temp.merge(df2, on=columns, how="left")
        new_columns = [column for column in list(df_temp) if column not in columns]
        df1 = pd.concat([df1, df_temp[new_columns]], axis=1)
        return df1

    def base_fea(self):
        for i in range(self.days_to_predict):
            prediction_d = self.number_of_train + (i + 1)
            self.df_sales_train_validation[f"d_{prediction_d}"] = np.nan

        df_sales_features = self.df_sales_train_validation.melt(
            id_vars=self.index_columns
            , var_name="d"
            , value_name="sales"
        )

        # Technics: converting strings to categorical variables
        for column in self.index_columns:
            df_sales_features[column] = df_sales_features[column].astype("category")

        df_sell_prices = self.prices_df

        # Create features
        # Items are available after that a certain time
        df_available_after = df_sell_prices.groupby(["store_id", "item_id"])["wm_yr_wk"].agg(["min"]).reset_index()
        df_available_after.columns = ["store_id", "item_id", "available_after"]

        # Join df_sales_features and df_available_after
        df_sales_features = self.merge_by_concat(df_sales_features, df_available_after, ["store_id", "item_id"])

        # We can drop those rows before available date
        # To achieve this, we need df_calendar's help
        df_calendar = self.calendar_df

        # Join df_sales_features and df_calendar
        df_sales_features = self.merge_by_concat(df_sales_features, df_calendar[["d", "wm_yr_wk"]], ["d"])

        # We only need those entries after "available_after"
        df_sales_features = df_sales_features[df_sales_features["wm_yr_wk"] >= df_sales_features["available_after"]]
        df_sales_features = df_sales_features.reset_index(drop=True)

        # Technics: we know the minimum of a certain column, so we find the difference between each row and its minimum
        # and store those differences in int16
        df_sales_features.drop(["wm_yr_wk"], axis=1, inplace=True)
        df_sales_features["available_after"] = (
                df_sales_features["available_after"] - df_sales_features["available_after"].min()).astype(np.int16)

        # Technics: for column "d", we would like to store it with int16 format
        df_sales_features["d"] = df_sales_features["d"].apply(lambda x: int(x[2:])).astype(np.int16)

        # Sort values to easily join features later
        df_sales_features.sort_values(by=["id", "d"], inplace=True)
        df_sales_features.reset_index(drop=True, inplace=True)

        # Save pickle file
        df_sales_features.to_pickle("./features/sales_basic_features.pkl")

    def calendar_fea(self):
        df_calendar = self.calendar_df
        # Select the necessary information
        # For example, we can extract day, month or year information from "date" column

        calendar_selected_columns = [
            "date"
            , "d"
            , "event_name_1"
            , "event_type_1"
            , "event_name_2"
            , "event_type_2"
            , "snap_CA"
            , "snap_TX"
            , "snap_WI"
        ]
        df_calendar_features = df_calendar[calendar_selected_columns]

        # Technics: converting strings to categorical variables
        calendar_category_columns = [
            "event_name_1"
            , "event_type_1"
            , "event_name_2"
            , "event_type_2"
            , "snap_CA"
            , "snap_TX"
            , "snap_WI"
        ]
        for column in calendar_category_columns:
            df_calendar_features[column] = df_calendar_features[column].astype("category")

        # Create features
        # Convert date to datetime variables and store the derivative information in int8
        df_calendar_features["date"] = pd.to_datetime(df_calendar_features["date"])
        df_calendar_features["day"] = df_calendar_features["date"].dt.day.astype(np.int8)
        df_calendar_features["weekday"] = df_calendar_features["date"].dt.dayofweek.astype(np.int8)
        df_calendar_features["week"] = df_calendar_features["date"].dt.week.astype(np.int8)
        df_calendar_features["month"] = df_calendar_features["date"].dt.month.astype(np.int8)
        df_calendar_features["year"] = (
                df_calendar_features["date"].dt.year - df_calendar_features["date"].dt.year.min()).astype(np.int8)
        df_calendar_features["week_of_month"] = df_calendar_features["date"].dt.day.apply(lambda x: ceil(x / 7)).astype(
            np.int8)
        df_calendar_features["is_weekend"] = (df_calendar_features["weekday"] >= 5).astype(np.int8)

        # Technics: for column "d", we would like to store it with int16 format
        df_calendar_features["d"] = df_calendar_features["d"].apply(lambda x: int(x[2:])).astype(np.int16)

        # Save pickle file
        df_calendar_features.to_pickle("./features/calendar_features.pkl")

    def prices_fea(self):
        # Load and check dataset
        df_sell_prices = self.prices_df

        # Create features
        # Selling prices are not as fluctuating as we expect,
        # so we only need several characteristics to capture their distribution
        df_sell_prices_grouped = df_sell_prices.groupby(["store_id", "item_id"])
        df_sell_prices["price_max"] = df_sell_prices_grouped["sell_price"].transform("max").astype(np.float16)
        df_sell_prices["price_min"] = df_sell_prices_grouped["sell_price"].transform("min").astype(np.float16)
        df_sell_prices["price_mean"] = df_sell_prices_grouped["sell_price"].transform("mean").astype(np.float16)
        df_sell_prices["price_std"] = df_sell_prices_grouped["sell_price"].transform("std").astype(np.float16)
        df_sell_prices["price_scaled"] = (
                (df_sell_prices["sell_price"] - df_sell_prices["price_min"])
                / (df_sell_prices["price_max"] - df_sell_prices["price_min"])
        ).astype(np.float16)
        df_sell_prices["price_nunique"] = df_sell_prices_grouped["sell_price"].transform("nunique").astype(np.int16)
        df_sell_prices["item_nunique"] = df_sell_prices.groupby(["store_id", "sell_price"])["item_id"].transform(
            "nunique").astype(np.int16)

        # Join df_sell_prices and raw df_calendar
        df_price_features = self.merge_by_concat(df_sell_prices, self.calendar_df[["wm_yr_wk", "month", "year", "d"]],
                                                 ["wm_yr_wk"])

        # Create features
        # Evaluate how do prices change periodically
        df_price_features["price_mean_change_week"] = (
                df_price_features["sell_price"] / df_price_features.groupby(["store_id", "item_id", "wm_yr_wk"])[
            "sell_price"].transform("mean")
        ).astype(np.float16)
        df_price_features["price_mean_change_month"] = (
                df_price_features["sell_price"] / df_price_features.groupby(["store_id", "item_id", "month"])[
            "sell_price"].transform("mean")
        ).astype(np.float16)
        df_price_features["price_mean_change_year"] = (
                df_price_features["sell_price"] / df_price_features.groupby(["store_id", "item_id", "year"])[
            "sell_price"].transform("mean")
        ).astype(np.float16)

        # Check dataset

        price_selected_columns = [
            "store_id"
            , "item_id"
            , "d"
            , "sell_price"
            , "price_max"
            , "price_min"
            , "price_mean"
            , "price_std"
            , "price_scaled"
            , "price_nunique"
            , "item_nunique"
            , "price_mean_change_week"
            , "price_mean_change_month"
            , "price_mean_change_year"
        ]
        df_price_features = df_price_features[price_selected_columns]

        # Technics: converting strings to categorical variables
        price_category_columns = ["store_id", "item_id"]
        for column in price_category_columns:
            df_price_features[column] = df_price_features[column].astype("category")

        # Technics: for column "sell_price", we would like to store it with float16 format
        df_price_features["sell_price"] = df_price_features["sell_price"].astype(np.float16)

        # Technics: for column "d", we would like to store it with int16 format
        df_price_features["d"] = df_price_features["d"].apply(lambda x: int(x[2:])).astype(np.int16)

        # Save pickle file
        df_price_features.to_pickle("./features/price_features.pkl")

    def lag_fea(self):
        df_lag_features = pd.read_pickle("./features/sales_basic_features.pkl")

        # Get necessary columns only
        df_lag_features = df_lag_features[["id", "d", "sales"]]

        # Generate basic lag features and control the memory usage
        df_lag_grouped = df_lag_features.groupby(["id"])["sales"]

        for i in range(self.days_to_predict):
            df_lag_features = df_lag_features.assign(
                **{f"sales_lag_{str(i + 1)}": df_lag_grouped.transform(lambda x: x.shift(i + 1))})
            df_lag_features[f"sales_lag_{str(i + 1)}"] = df_lag_features[f"sales_lag_{str(i + 1)}"].astype(np.float16)

        # Save pickle file
        df_lag_features.to_pickle("./features/sales_lag_features.pkl")

    def rolling_lag_fea(self):
        # Load dataset from our previous work
        df_rolling_features = pd.read_pickle("./features/sales_basic_features.pkl")

        # Get necessary columns only
        df_rolling_features = df_rolling_features[["id", "d", "sales"]]

        # Create features
        # Generate rolling lag features and control the memory usage

        df_rolling_grouped = df_rolling_features.groupby(["id"])["sales"]

        for day in self.rolling_days:
            df_rolling_features[f"rolling_{str(day)}_max"] = df_rolling_grouped.transform(
                lambda x: x.shift(self.days_to_predict).rolling(day).max()).astype(np.float16)
            df_rolling_features[f"rolling_{str(day)}_min"] = df_rolling_grouped.transform(
                lambda x: x.shift(self.days_to_predict).rolling(day).min()).astype(np.float16)
            df_rolling_features[f"rolling_{str(day)}_median"] = df_rolling_grouped.transform(
                lambda x: x.shift(self.days_to_predict).rolling(day).median()).astype(np.float16)
            df_rolling_features[f"rolling_{str(day)}_mean"] = df_rolling_grouped.transform(
                lambda x: x.shift(self.days_to_predict).rolling(day).mean()).astype(np.float16)
            df_rolling_features[f"rolling_{str(day)}_std"] = df_rolling_grouped.transform(
                lambda x: x.shift(self.days_to_predict).rolling(day).std()).astype(np.float16)

        # Save pickle file
        df_rolling_features.to_pickle("./features/sales_rolling_features.pkl")


if __name__ == '__main__':
    data_dir = './data/'
    feature_dir = './features/'

    # feature preparing
    fea_pre = FeaturePre(data_dir, feature_dir)
    fea_pre.base_fea()
    fea_pre.prices_fea()
    fea_pre.calendar_fea()
    fea_pre.lag_fea()
    fea_pre.rolling_lag_fea()
