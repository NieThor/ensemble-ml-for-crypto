import pandas as pd
import numpy as np

import time

import numba


@numba.njit()
def cum_li(pct_change, volume, lim):
    total_pct = 0.
    total_vol = 0.
    result_pct = np.empty_like(pct_change)
    result_vol = np.empty_like(volume)
    group = np.empty_like(volume)
    group_nr = 0
    for i, y in enumerate(pct_change):
        if total_pct > lim:
            total_pct = 0.
            total_vol = 0.
            group_nr += 1
        total_pct += y
        total_vol += volume[i]
        result_pct[i] = total_pct
        result_vol[i] = total_vol
        group[i] = group_nr
    return result_pct, result_vol, group


def resample_data(df: pd.DataFrame, resample_type=None):
    df['pct_change'] = df['close'].pct_change().abs()
    df['pct_change'].fillna(0, inplace=True)
    resample_threshold = 0.02
    df['cum_sum_w_reset'], df['vol'], df['group'] = cum_li(df['pct_change'].values, df['volume'].values, resample_threshold)

    resampled_df = df.groupby(['group']).agg({'open': 'first', 'high': 'max','low': 'min', 'close': 'last', 'vol': 'last'})
    resampled_df.rename(columns={'vol': 'volume'}, inplace=True)
    resampled_df = resampled_df[1:]
    return resampled_df
