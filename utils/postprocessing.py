import numpy as np
import pandas as pd


whales_array = np.load('D:/IdeaProjects/whales/embeddings.npy')
whales = pd.DataFrame(data=whales_array)
whales_csv = pd.read_csv('D:/IdeaProjects/whales/data/train.csv')
big_whales = pd.concat([whales, whales_csv], axis=1)
big_whales = big_whales.drop(['Image'], axis=1)
whale_means = big_whales.groupby(['Id']).mean().reset_index()

print(whale_means)

whale_means.to_csv('whales.csv', index=False)


