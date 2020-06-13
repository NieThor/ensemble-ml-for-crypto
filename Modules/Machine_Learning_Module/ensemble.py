import numpy as np


def ensemble(predictions: np.ndarray):
    ensembled_predictions = predictions.sum(axis=0) / predictions.shape[0]
    ensembled_predictions = [1 if ensembled_predictions[i] > 0.5 else 0 for i in range(len(ensembled_predictions))]

    return ensembled_predictions
