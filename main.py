import pandas as pd

from Modules.Machine_Learning_Module.data_preparation_ml import prepare_data
from Modules.Machine_Learning_Module.model_fitting import fit_models
from Modules.Machine_Learning_Module.prediction import predict
from Modules.data_preparation import resample_data
from Modules.indicators import apply_indicators


def load_data():
    df = pd.DataFrame()
    return df


def main():
    df = load_data()
    validation_df = df[-50:]
    prepared_df = resample_data(df[:-50])
    df = apply_indicators(prepared_df)
    train, test = prepare_data(df)
    models = fit_models(train, test)
    predictions = predict(validation_df, models)
    print(predictions)


if __name__ == '__main__':
    main()
