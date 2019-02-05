from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
from time import strftime, gmtime

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.manifold import TSNE

from model.resnet import resnet_like_33, resnet_like_36
from model.shallow import mnist_5
from loss.triplet_loss import triplet_loss
from utils.sequence import WhalesSequence

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))


class Siamese(object):
    def __init__(self, model, strategy='batch_all', input_shape=(384, 512, 3), embedding_size=128):
        self.strategy = strategy
        self.input_shape = input_shape
        self.embedding_size = embedding_size

        self.embeddings = None
        self.predictions = None

        self.cache_dir = os.path.join('cache', strftime("cache-%y%m%d-%H%M%S", gmtime()))
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.isdir(os.path.join(self.cache_dir, 'training')):
            os.makedirs(os.path.join(self.cache_dir, 'training'))
        if not os.path.isdir(os.path.join(self.cache_dir, 'debug')):
            os.makedirs(os.path.join(self.cache_dir, 'debug'))
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

    def train(self, csv, img_dir, epochs=10, batch_size=10, learning_rate=0.001, margin=0.5):
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate), loss=triplet_loss(margin, self.strategy))

        whales_data = self._read_csv(csv)
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], y_set=whales_data[:, 1], batch_size=batch_size)
        self.model.fit_generator(whales,
                                 epochs=epochs,
                                 callbacks=[ModelCheckpoint(filepath=os.path.join(self.cache_dir, 'training', 'checkpoint-{epoch:02d}.h5'), save_weights_only=True)])
        self.model.save(os.path.join(self.cache_dir, 'final_model.h5'))
        self.save_weights(os.path.join(self.cache_dir, 'final_weights.h5'))

    def make_embeddings(self, csv, img_dir, batch_size=10):
        whales_data = self._read_csv(csv)
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], batch_size=batch_size)
        pred = self.model.predict_generator(whales, verbose=1)

        whales_df = pd.DataFrame(data=whales_data, columns=['Image', 'Id'])
        pred_df = pd.DataFrame(data=pred)
        pred_df = pd.concat([pred_df, whales_df], axis=1)
        pred_df = pred_df.drop(['Image'], axis=1)
        self.embeddings = pred_df.groupby(['Id']).mean().reset_index()
        self.save_embeddings(os.path.join(self.cache_dir, 'embeddings.pkl'))

    def predict(self, img_dir):
        assert self.embeddings is not None
        whales_data = np.array(os.listdir(img_dir))
        whales_seq = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data, batch_size=1)
        whales = self.model.predict_generator(whales_seq, verbose=1)

        np.save(os.path.join(self.cache_dir, 'debug', 'raw_predictions'), whales)

        embeddings = self.embeddings.drop(['Id'], axis=1)
        concat = np.concatenate((embeddings, whales), axis=0)
        prod = np.dot(concat, np.transpose(concat))
        sq_norms = np.reshape(np.diag(prod), (-1, 1))

        dist = sq_norms - 2.0 * prod + np.transpose(sq_norms)
        dist = np.maximum(dist, 0.0)
        dist = dist[self.embeddings.shape[0]:, :self.embeddings.shape[0]]

        predictions = np.apply_along_axis(np.argpartition, 1, dist, 5)
        self.predictions = pd.DataFrame(data=predictions[:, :5])
        self.predictions = pd.concat([pd.DataFrame(data=whales_data), self.predictions], axis=1)
        self.predictions.columns = ['Image'] + list(range(5))
        self.save_predictions(os.path.join(self.cache_dir, 'predictions.pkl'))

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename, by_name=True)

    def save_embeddings(self, filename):
        self.embeddings.to_pickle(filename)

    def load_embeddings(self, filename):
        self.embeddings = pd.read_pickle(filename)

    def save_predictions(self, filename):
        self.predictions.to_pickle(filename)

    def load_predictions(self, filename):
        self.predictions = pd.read_pickle(filename)

    def save_state(self, filename):
        pass

    def load_state(self, filename):
        pass

    def _read_csv(self, csv):
        csv_data = pd.read_csv(csv)
        whales = np.sort(csv_data['Id'].unique())
        mapping = {}
        reverse_mapping = {}
        for i, w in enumerate(whales):
            mapping[w] = i
            reverse_mapping[i] = [w]
        data = csv_data.replace({'Id': mapping})
        np.save(os.path.join(self.cache_dir, 'whales_to_idx_mapping'), mapping)
        np.save(os.path.join(self.cache_dir, 'idx_to_whales_mapping'), reverse_mapping)
        return data.values

    @staticmethod
    def draw_tsne(vectors):
        tsne = TSNE(n_components=3, verbose=1, n_iter=300).fit_transform(vectors)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2])
        plt.show()

    def make_kaggle_csv(self, mapping_file):
        mapping = np.load(mapping_file).item()
        for k in mapping:
            mapping[k] = mapping[k][0]
        predictions = self.predictions.replace({0: mapping, 1: mapping, 2: mapping, 3: mapping, 4: mapping})
        predictions['Id'] = predictions[0] + ' ' + predictions[1] + ' ' + predictions[2] + ' ' + predictions[3] + ' ' + predictions[4]
        predictions.to_csv(os.path.join(self.cache_dir, 'submission.csv'), index=False, columns=['Image', 'Id'])









