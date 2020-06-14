import pandas as pd
import numpy as np


def prepare_data(df: pd.DataFrame):
    df = feature_scale(df)
    df = get_target_data(df)

    return df


def get_target_data(df: pd.DataFrame):
    fee = 0.00075
    df['target'] = np.nan

    df['target'][df['close'].shift(1) * (1 + fee) < df['close'] * (1 - fee)] = 0
    df['target'][df['close'].shift(1) * (1 + fee) > df['close'] * (1 - fee)] = 1
    df = df[200:]
    while df['target'].isnull().values.any():
        df['target'][df['close'].shift(1) * (1 + fee) == df['close'] * (1 - fee)] = \
            df['target'].shift(1)[df['close'].shift(1) * (1 + fee) == df['close'] * (1 - fee)]
    return df


def feature_scale(df: pd.DataFrame):
    # normalize dataframe
    df = (df - df.mean()) / df.std()

    return df

