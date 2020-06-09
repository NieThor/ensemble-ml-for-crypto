import pandas as pd

from Modules.Machine_Learning_Module.data_preparation_ml import prepare_data
from Modules.Machine_Learning_Module.model_fitting import fit_models
from Modules.Machine_Learning_Module.prediction import predict
from Modules.data_preparation import resample_data
from Modules.indicators import apply_indicators

import os


def load_data():
    path = '.\\data\\binance'
    data = {}
    time_frame = '5m'
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(f'{time_frame}.json'):
            ticker = filename.split(f'-{time_frame}')[0].replace('_', '/')
            data[ticker] = pd.read_json(os.path.join(path, filename))
            data[ticker].columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            data[ticker].set_index('time', inplace=True)
            if len(data) > 1:
                break
            continue
        else:
            continue
    return data


def main():
    data = load_data()
    validation_data = {key: df[-50:] for key, df in data.items()}
    prepared_data = {key: resample_data(df[:-50]) for key, df in data.items()}
    data = {key: apply_indicators(df) for key, df in prepared_data.items()}
    train_data, test_data = {}, {}
    for key, df in data.items():
        train_data[key], test_data[key] = prepare_data(df)
    models = {key: fit_models(train_data[key], test_data[key]) for key in train_data.keys()}
    predictions = {key: predict(validation_df, models[key]) for key, validation_df in validation_data.items()}
    print(predictions)


if __name__ == '__main__':
    main()
