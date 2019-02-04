import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

embeddings = pd.DataFrame(data=np.load('D:/IdeaProjects/whales/embeddings.npy'), columns=(['Id'] + (list(range(128)))))
embeddings = embeddings.drop(columns=['Id'])
tsne = TSNE(n_components=3, verbose=1, n_iter=300).fit_transform(embeddings.values)
print(tsne.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2])
plt.show()
# pred = np.load('D:/IdeaProjects/whales/predictions.npy')
# print(pred[:, :5])
#
# csv_data = pd.read_csv('D:/IdeaProjects/whales/data/train.csv')
# whales = np.sort(csv_data['Id'].unique())
# mapping = {}
# for i, w in enumerate(whales):
#     mapping[i] = w
#
#
# def idx_to_whale(idx):
#     return mapping[idx]
#
#
# v_idx_to_whale = np.vectorize(idx_to_whale)
#
# data = v_idx_to_whale(pred)
#
# print(data)





