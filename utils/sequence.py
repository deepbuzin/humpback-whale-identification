from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.utils import Sequence

from .preprocessing import fetch, resize, pad


class WhalesSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([self.preprocess(fetch(name)) for name in batch_x]), np.array(batch_y)

    def preprocess(self, img):
        assert len(img.shape) == 3
        h, w, _ = img.shape
        if h / w <= 0.75:
            img = resize(img, (1024, int(1024 * h / w)))
        else:
            img = resize(img, (int(768 * w / h), 768))
        img = pad(img(1024, 768))
        return img



