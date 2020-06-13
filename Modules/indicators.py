import talib.abstract as ta
import qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd


def ichimoku(df: pd.DataFrame, cl=20, bl=60, ls=120, dp=30):
    nine_period_high = df['high'].rolling(window=cl).max()
    nine_period_low = df['low'].rolling(window=cl).min()
    df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line): (bl-period high + bl-period low)/2))
    period_bl_high = df['high'].rolling(window=bl).max()
    period_bl_low = df['low'].rolling(window=bl).min()
    df['kijun_sen'] = (period_bl_high + period_bl_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(dp)

    # Senkou Span B (Leading Span B): (ls-period high + ls-period low)/2))
    period_ls_high = df['high'].rolling(window=ls).max()
    period_ls_low = df['low'].rolling(window=ls).min()
    df['senkou_span_b'] = ((period_ls_high + period_ls_low) / 2).shift(dp)

    # The most current closing price plotted dp time periods behind (optional)
    # df['chikou_span'] = df['close'].shift(-dp)  #
    return df


def apply_indicators(df: pd.DataFrame):

    # ADX
    df['adx'] = ta.ADX(df)

    # EMA
    df['ema_5'] = ta.EMA(df, 5)
    df['ema_10'] = ta.EMA(df, 10)
    df['ema_20'] = ta.EMA(df, 20)
    df['ema_50'] = ta.EMA(df, 50)
    df['ema_100'] = ta.EMA(df, 100)
    df['ema_200'] = ta.EMA(df, 200)


    # MACD
    macd = ta.MACD(df)
    df['macd'] = macd['macd']
    df['macdsignal'] = macd['macdsignal']
    df['macdhist'] = macd['macdhist']

    # inverse Fisher rsi/ RSI
    df['rsi'] = ta.RSI(df)
    rsi = 0.1 - (df['rsi']-50)
    df['i_rsi'] = (np.exp(2 * rsi)-1)/(np.exp(2 * rsi)+1)

    # Stoch fast
    stoch_fast = ta.STOCHF(df)
    df['fastd'] = stoch_fast['fastd']
    df['fastk'] = stoch_fast['fastk']

    # Stock slow
    stoch_slow = ta.STOCH(df)
    df['slowd'] = stoch_slow['slowd']
    df['slowk'] = stoch_slow['slowk']

    # Bollinger bands
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df['bb_lowerband'] = bollinger['lower']
    df['bb_middleband'] = bollinger['mid']
    df['bb_upperband'] = bollinger['upper']

    # ROC
    df['roc'] = ta.ROC(df, 10)

    # CCI
    df['cci'] = ta.CCI(df, 14)

    # on balance volume
    df['obv'] = ta.OBV(df)

    # Average True Range
    df['atr'] = ta.ATR(df, 14)



    df = ichimoku(df)

    return df
