from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import os

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from model.resnet import resnet_like_33, resnet_like_36
from model.shallow import mnist_5
from loss.triplet_loss import triplet_loss
from utils.sequence import WhalesSequence


class Siamese(object):
    def __init__(self, model, strategy='batch_all', input_shape=(384, 512, 3), embedding_size=128):
        self.strategy = strategy
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
        elif model_name == 'shallow_mnist':
            return mnist_5(input_shape=input_shape, embedding_size=embedding_size)
        else:
            raise ValueError('no such model: %s' % model_name)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, weights):
        self.model.load_weights(weights, by_name=True)

    def train(self, csv, img_dir, epochs=10, batch_size=10, learning_rate=0.001, margin=0.5):
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate), loss=triplet_loss(margin, self.strategy))

        whales_data = Siamese._read_csv(csv)
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], y_set=whales_data[:, 1], batch_size=batch_size)
        self.model.fit_generator(whales, epochs=epochs, callbacks=[ModelCheckpoint(filepath='checkpoint-{epoch:02d}', save_weights_only=True)])
        self.model.save('autosave_model.h5')
        self.save_weights('autosave_weights.h5')

    def make_embeddings(self, csv, img_dir, batch_size=10):
        whales_data = Siamese._read_csv(csv)
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], batch_size=batch_size)
        pred = self.model.predict_generator(whales, verbose=1)

        pred_df = pd.DataFrame(data=pred)
        pred_df = pd.concat([pred_df, pd.DataFrame(data=whales_data)], axis=1)
        pred_df = pred_df.drop(['Image'], axis=1)
        self.embeddings = pred_df.groupby(['Id']).mean().reset_index()
        np.save('embeddings', self.embeddings)

    def predict(self, img_dir):
        assert self.embeddings is not None
        whales_data = np.array(os.listdir(img_dir))
        whales_seq = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data, batch_size=1)
        whales = self.model.predict_generator(whales_seq, verbose=1)

        concat = np.concatenate((self.embeddings, whales), axis=0)
        prod = np.dot(concat, np.transpose(concat))
        sq_norms = np.reshape(np.diag(prod), (-1, 1))

        dist = sq_norms - 2.0 * prod + np.transpose(sq_norms)
        dist = np.maximum(dist, 0.0)
        dist = dist[self.embeddings.shape[0]:, :self.embeddings.shape[0]]

        predictions = np.apply_along_axis(np.argpartition, 0, dist, 5)
        np.save('predictions', predictions)

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







