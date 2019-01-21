from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

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

        whales_data = np.genfromtxt(csv, dtype=str, delimiter=',', skip_header=True)
        # TODO check if labels are replaced with numbers
        # TODO alternatively can just work with raw whales dataset
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], y_set=whales_data[:, 1], batch_size=batch_size)
        self.model.fit_generator(whales, epochs=epochs, callbacks=[ModelCheckpoint(filepath='checkpoint-{epoch:02d}')])
        self.model.save('autosave_model.h5')

    def make_embeddings(self, img_dir, batch_size=10):
        whales_data = np.array(os.listdir(img_dir))
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data, batch_size=batch_size)
        self.embeddings = self.model.predict_generator(whales, verbose=1)
        # TODO make postprocessing

    def save_embeddings(self, filename):
        np.save(filename, self.embeddings)

    def load_embeddings(self, filename):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


def fit(model, img_dir, csv):
    data = np.genfromtxt(csv, dtype=str, delimiter=',', skip_header=True)
    train = WhalesSequence(img_dir, input_shape=(384, 512, 3), x_set=data[:, 0], y_set=data[:, 1], batch_size=10)
    model.fit_generator(train, epochs=10, callbacks=[ModelCheckpoint(filepath='checkpoint-{epoch:02d}')])
    model.save('model_777.h5')


def make_embeddings(model, img_dir):
    data = np.array(os.listdir(img_dir))
    train = WhalesSequence(img_dir, input_shape=(384, 512, 3), x_set=data, batch_size=10)
    emb = model.predict_generator(train, verbose=1)
    print(emb)
    np.save('np_embeddings.pkl', emb)


def predict(model, img_dir):
    data = np.array(os.listdir(img_dir))
    test = WhalesSequence(img_dir, input_shape=(384, 512, 3), x_set=data, batch_size=10)
    emb = model.predict_generator(test, verbose=1)


if __name__ == '__main__':
    m = resnet_like_33(input_shape=(384, 512, 3), embedding_size=128)
    m.compile(optimizer=Adam(0.0001), loss=triplet_loss(margin=0.5, strategy='batch_all'))
    fit(m, 'D:/IdeaProjects/whales/data/train', 'D:/IdeaProjects/whales/data/train_fixed.csv')

    # m_trained = load_model('model.h5', custom_objects={'batch_hard': triplet_loss_batch_hard(margin=0.2)})
    # make_embeddings(m_trained, 'D:/IdeaProjects/whales/data/train')



