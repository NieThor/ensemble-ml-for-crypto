import numpy as np


def ensemble(predictions):
    ensembled_predictions = [np.sum(predictions[i])/(len(predictions[i])) for i in range(len(predictions))]

    return ensembled_predictions
