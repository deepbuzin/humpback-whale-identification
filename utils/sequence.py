from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.utils import Sequence

from .preprocessing import fetch, resize, pad


class WhalesSequence(Sequence):
    def __init__(self, img_dir, input_shape, x_set, y_set=None, batch_size=16):
        self.x, self.y = x_set, y_set
        self.img_dir = img_dir
        self.input_shape = input_shape
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.y is None:
            return np.array([self.preprocess(fetch(self.img_dir, name)) for name in batch_x])

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
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



