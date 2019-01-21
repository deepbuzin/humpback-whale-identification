import numpy as np
import pandas as pd


def process(predictions):
    whales = pd.read_csv('whales.csv')
    whales.sort_values(['Id'])
    whales = whales.drop(['Id'], axis=1).values

    whales_preds = np.concatenate((whales, predictions), axis=0)

    prod = np.dot(whales_preds, np.transpose(whales_preds))
    sq_norms = np.reshape(np.diag(prod), (-1, 1))

    dist = sq_norms - 2.0*prod + np.transpose(sq_norms)
    dist = np.maximum(dist, 0.0)

    print(dist.shape)


if __name__ == '__main__':
    process(np.zeros((8000, 192)))









