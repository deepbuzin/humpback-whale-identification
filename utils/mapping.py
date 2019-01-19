import numpy as np
import pandas as pd


def extract_id_mapping(data):
    whales = np.sort(data['Id'].unique())
    mapping = {}
    for i, w in enumerate(whales):
        mapping[w] = i
    return mapping

if __name__ == '__main__':
    data = pd.read_csv('D:/IdeaProjects/whales/data/train.csv')
    mapping = extract_id_mapping(data)
    data = data.replace({'Id': mapping})
    data.to_csv('D:/IdeaProjects/whales/data/train_fixed.csv', index=False)
