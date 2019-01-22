from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from model.resnet import resnet_like_33, resnet_like_36
from loss.triplet_loss import triplet_loss
from utils.sequence import WhalesSequence


class Siamese(object):
    def __init__(self, model, input_shape=(384, 512, 3), embedding_size=128):
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.embeddings = None
        self.model = Siamese.build_model(model, input_shape, embedding_size)

    @staticmethod
    def build_model(model_name, input_shape, embedding_size):
        if model_name == 'resnet_like_33':
            return resnet_like_33(input_shape=input_shape, embedding_size=embedding_size)
        elif model_name == 'resnet_like_36':
            return resnet_like_36(input_shape=input_shape, embedding_size=embedding_size)
        else:
            raise ValueError('no such model: %s' % model_name)

    def train(self, csv, img_dir, epochs=10, batch_size=10, learning_rate=0.001, margin=0.5, strategy='batch_all'):
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate), loss=triplet_loss(margin, strategy))

        whales_data = Siamese._read_csv(csv)
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], y_set=whales_data[:, 1], batch_size=batch_size)
        self.model.fit_generator(whales, epochs=epochs, callbacks=[ModelCheckpoint(filepath='checkpoint-{epoch:02d}')])
        self.model.save('autosave_model.h5')

    def make_embeddings(self, csv, img_dir, batch_size=10):
        whales_data = Siamese._read_csv(csv)
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], batch_size=batch_size)
        pred = self.model.predict_generator(whales, verbose=1)

        pred_df = pd.DataFrame(data=pred)
        pred_df = pd.concat([pred_df, pd.DataFrame(data=whales_data)], axis=1)
        pred_df = pred_df.drop(['Image'], axis=1)
        self.embeddings = pred_df.groupby(['Id']).mean().reset_index()

    def predict(self):
        pass

    def save_embeddings(self, filename):
        np.save(filename, self.embeddings)

    def load_embeddings(self, filename):
        self.embeddings = np.load(filename)

    @staticmethod
    def _read_csv(csv):
        csv_data = pd.read_csv(csv)
        whales = np.sort(csv_data['Id'].unique())
        mapping = {}
        for i, w in enumerate(whales):
            mapping[w] = i
        data = csv_data.replace({'Id': mapping})
        return data.values

    def save_model(self):
        pass

    def load_model(self):
        pass

    def save_weights(self):
        pass

    def load_weights(self):
        pass





