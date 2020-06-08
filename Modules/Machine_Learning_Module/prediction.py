import pandas as pd

from Modules.Machine_Learning_Module.ensemble import ensemble


def predict(df: pd.DataFrame, models):
    predictions = []
    for i, model in enumerate(models):
        predictions[i] = model.predict(df)
    ensembled_predictions = ensemble(predictions)
    return ensembled_predictions
