import pandas as pd

import time


def resample_data(df: pd.DataFrame, resample_type=None):
    resampled_df = pd.DataFrame(data=None, columns=df.columns)
    df['pct_change'] = df['close'].pct_change().abs()
    resample_threshold = 0.02
    last_value = 0
    low_ = df['low'].iloc[1]
    high_ = df['high'].iloc[1]
    open_ = df['open'].iloc[1]
    close_ = df['close'].iloc[1]
    volume_ = 0
    for i, row in enumerate(df.iterrows()):
        row = row[1]
        if i != 0:
            curr_value = row['pct_change']
            if last_value < resample_threshold:
                curr_value += last_value
                low_ = row['low'] if row['low'] < low_ else low_
                high_ = row['high'] if row['high'] > high_ else high_
                close_ = row['close']
                volume_ \
                    += row['volume']
            else:
                resampled_df = resampled_df.append({'open': open_,
                                                    'high': high_,
                                                    'low': low_,
                                                    'close': close_,
                                                    'volume': volume_}, ignore_index=True)
                low_ = row['low']
                high_ = row['high']
                open_ = row['open']
                close_ = row['close']
                volume_ = row['volume']

            last_value = curr_value

    return resampled_df
