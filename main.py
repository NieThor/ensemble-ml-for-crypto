import pandas as pd

from Modules.Machine_Learning_Module.data_preparation_ml import prepare_data
from Modules.Machine_Learning_Module.model_fitting import fit_models
from Modules.data_preparation import resample_data
from Modules.indicators import apply_indicators

import os
import time

import pickle


def do_it(df, key):
    start = time.time()
    df = resample_data(df)
    print(f'Resampling data of {key} took {time.time() - start}s.')
    start = time.time()
    df = apply_indicators(df)
    print(f'Applying indicators to {key} took {time.time() - start}s.')
    start = time.time()
    df = prepare_data(df)
    print(f'Preparing data for modelfitting of {key} took {time.time() - start}s.')
    start = time.time()
    models = fit_models(df)
    print(f'The fitting of {key} took {(time.time() - start)}s with {len(df)} points of data.')
    start = time.time()
    print(f'The predictions of {key} took {(time.time() - start)}s.')
    with open(f'models/{key.replace("/", "_")}.pkl', 'wb') as file:
        pickle.dump(models, file)


def main():
    path = '.\\data\\binance'
    time_frame = '1m'
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(f'{time_frame}.json'):
            start = time.time()
            ticker = filename.split(f'-{time_frame}')[0].replace('_', '/')
            df = pd.read_json(os.path.join(path, filename))
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df.set_index('time', inplace=True)
            print(f'The loading of {ticker} data took {(time.time() - start)}s with {len(df)} points of data.')
            do_it(df, ticker)
            continue
        else:
            continue


if __name__ == '__main__':
    main()
