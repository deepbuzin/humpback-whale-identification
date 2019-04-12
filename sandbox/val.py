from __future__ import print_function

import pandas as pd
import numpy as np

#val = pd.read_csv('val.csv')
val = pd.read_csv('train.csv')
true_labels = val["Id"].values

pred = pd.read_pickle('trained/predictions.pkl')
inds = np.unique(true_labels, return_index=True)[1]
whale_name = [true_labels[ind] for ind in sorted(inds)]
whale_id = list(range(1, len(whale_name)+1))
mapping = dict(zip(whale_id, whale_name))
pred = pred.replace({0: mapping, 1: mapping, 2: mapping, 3: mapping, 4: mapping})
pred_labels = pred[0].values

print('true labels: \n', true_labels)
print('pred labels: \n', pred_labels)

acc = sum(true_labels == pred_labels) / len(true_labels)
print('accuracy: ', acc)