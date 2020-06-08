import pandas as pd


def prepare_data(df: pd.DataFrame):
    scaled_df = feature_scale(df)
    train, test = split_dataset(scaled_df)

    return train, test


def feature_scale(df: pd.DataFrame):
    scaled_df = df.copy()
    return scaled_df


def split_dataset(df: pd.DataFrame):
    train = df
    test = df
    return train, test
