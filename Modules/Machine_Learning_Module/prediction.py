import pandas as pd
import numpy as np

from Modules.Machine_Learning_Module.ensemble import ensemble

from sklearn.metrics import accuracy_score


def predict(df: pd.DataFrame, models: []):
    predictions = np.ndarray(shape=(4, len(df)))
    y = df['target']
    df.drop(columns=['target'], inplace=True)
    for i, model in enumerate(models):
        predictions[i] = model.predict_proba(df)[:, 1]
    ensembled_predictions = ensemble(predictions)
    print(f'accuracy_score: {accuracy_score(y_pred=ensembled_predictions, y_true=y)}')
    return ensembled_predictions
