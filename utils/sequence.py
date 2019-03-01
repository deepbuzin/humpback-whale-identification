from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from keras.utils import Sequence
from sklearn.utils import shuffle

from .preprocessing import fetch, resize, pad


class WhalesSequence(Sequence):
    def __init__(self, img_dir, input_shape, x_set, y_set=None, batch_size=16):
        if y_set is not None:
            self.x, self.y = shuffle(x_set, y_set, random_state=666)
            self.classes, self.classes_count = np.unique(y_set, return_counts=True)
            self.dataset = pd.DataFrame(data={'x': x_set, 'y': y_set, 'used': np.zeros_like(y_set)})
            self.dataset['class_count'] = self.dataset.groupby('y')['y'].transform('count')
        else:
            self.x, self.y = x_set, None
            
        self.img_dir = img_dir
        self.input_shape = input_shape
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.y is None:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array([self.preprocess(fetch(self.img_dir, name)) for name in batch_x])

        unused = self.dataset.loc[self.dataset['used'] == 0]
        if len(unused.loc[self.dataset['class_count'] > 1]) >= 15 and len(unused.loc[self.dataset['class_count'] == 1]) >= 5:
            good = unused.loc[self.dataset['class_count'] > 1]['y'].sample(n=15)
            bad = unused.loc[self.dataset['class_count'] == 1]['y'].sample(n=5)
            include_classes = pd.concat([good, bad], axis=0).unique()
        else:
            include_classes = unused['y'].unique()

        if len(unused.loc[self.dataset['y'].isin(include_classes)]) >= self.batch_size:
            batch_indices = unused.loc[(self.dataset['y'].isin(include_classes))].sample(n=self.batch_size).index
        elif len(unused) >= self.batch_size:
            batch_indices = unused.sample(n=self.batch_size).index
        else:
            batch_indices = unused.sample(n=self.batch_size, replace=True).index

        self.dataset.loc[batch_indices, 'used'] = 1
        batch_x = self.dataset.iloc[batch_indices]['x'].values
        batch_y = self.dataset.iloc[batch_indices]['y'].values
        return np.array([self.preprocess(fetch(self.img_dir, name)) for name in batch_x]), np.array(batch_y)

    def preprocess(self, img):
        assert len(img.shape) == 3
        h, w, _ = img.shape
        if h / w <= 0.75:
            img = resize(img, (self.input_shape[1], int(self.input_shape[1] * h / w)))
        else:
            img = resize(img, (int(self.input_shape[0] * w / h), self.input_shape[0]))
        img = pad(img, (self.input_shape[1], self.input_shape[0]))
        return img

    def on_epoch_end(self):
        if self.y is not None:
            self.x, self.y = shuffle(self.x, self.y, random_state=666)
            self.classes, self.classes_count = np.unique(self.y, return_counts=True)
            self.dataset = pd.DataFrame(data={'x': self.x, 'y': self.y, 'used': np.zeros_like(self.y)})
            self.dataset['class_count'] = self.dataset.groupby('y')['y'].transform('count')

