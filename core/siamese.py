from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import os
import shutil
import json
from time import strftime, gmtime

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils.callbacks import TensorBoard
from sklearn.manifold import TSNE

from model.mobilenet import mobilenet_like
from model.resnet import resnet_like_33, resnet_like_36
from model.shallow import mnist_5
from model.dummy import dummy
from loss.triplet_loss import triplet_loss, soft_margin_triplet_loss
from utils.sequence import WhalesSequence

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))


class Siamese(object):
    def __init__(self, model, strategy='batch_all', input_shape=(384, 512, 3), embedding_size=128):
        self.model_name = model
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

        self.write_config()

    @staticmethod
    def build_model(model_name, input_shape, embedding_size):
        if model_name == 'mobilenet_like':
            return mobilenet_like(input_shape=input_shape, embedding_size=embedding_size, train_hidden_layers=True)
        elif model_name == 'resnet_like_33':
            return resnet_like_33(input_shape=input_shape, embedding_size=embedding_size)
        elif model_name == 'resnet_like_36':
            return resnet_like_36(input_shape=input_shape, embedding_size=embedding_size)
        elif model_name == 'shallow_mnist':
            return mnist_5(input_shape=input_shape, embedding_size=embedding_size)
        elif model_name == 'dummy':
            return dummy(input_shape=input_shape, embedding_size=embedding_size)
        else:
            raise ValueError('no such model: %s' % model_name)

    @staticmethod
    def get_vis_data(meta_dir):
        names = os.listdir(os.path.join(meta_dir, 'img'))
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(meta_dir, 'img', name)), cv2.COLOR_BGR2RGB) for name in names]
        return np.array(imgs).reshape((-1, 672, 896, 3))

    def train(self, csv, img_dir, meta_dir, epochs=10, batch_size=10, learning_rate=0.001, margin=0.5):
        self.model.summary()
        # self.model.compile(optimizer=Adam(learning_rate), loss=triplet_loss(margin, self.strategy))
        self.model.compile(optimizer=Adam(learning_rate), loss=soft_margin_triplet_loss)

        whales_data = self._read_csv(csv, mappings_filename=os.path.join(meta_dir, 'whales_to_idx_mapping.npy'))
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], y_set=whales_data[:, 1], batch_size=batch_size)
        self.model.fit_generator(whales,
                                 shuffle=True,
                                 epochs=epochs,
                                 callbacks=[ModelCheckpoint(filepath=os.path.join(self.cache_dir, 'training', 'checkpoint-{epoch:02d}.h5'), save_weights_only=True),
                                            TensorBoard(update_freq='epoch',
                                                        # embeddings_freq=1,
                                                        # embeddings_data=self.get_vis_data(meta_dir),
                                                        # embeddings_metadata=os.path.join(meta_dir, 'metadata.tsv'),
                                                        # embeddings_sprite=os.path.join(meta_dir, 'sprite.png'),
                                                        # embeddings_sprite_single_image_size=(50, 50),
                                                        embeddings_layer_names=['embeddings'],
                                                        log_dir=os.path.join(self.cache_dir, 'tensorboard_logs'))])
        self.model.save(os.path.join(self.cache_dir, 'final_model.h5'))
        self.save_weights(os.path.join(self.cache_dir, 'final_weights.h5'))

    def make_embeddings(self, img_dir, csv, batch_size=10):
        whales_data = self._read_csv(csv)
        whales = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data[:, 0], batch_size=batch_size)
        pred = self.model.predict_generator(whales, verbose=1)

        whales_df = pd.DataFrame(data=whales_data, columns=['Image', 'Id'])
        pred_df = pd.DataFrame(data=pred)
        pred_df = pd.concat([pred_df, whales_df], axis=1)
        pred_df = pred_df.drop(['Image'], axis=1)
        #pred_df = pred_df.groupby(['Id']).mean().reset_index()
        self.embeddings = pred_df.sort_values(by=['Id'])
        self.save_embeddings(os.path.join(self.cache_dir, 'embeddings.pkl'))

    def predict(self, img_dir, csv=''):
        assert self.embeddings is not None
        whales_data = np.array(os.listdir(img_dir)) if (csv == '') else self._read_csv(csv)[:, 0]
        whales_seq = WhalesSequence(img_dir, input_shape=self.input_shape, x_set=whales_data, batch_size=1)
        whales = self.model.predict_generator(whales_seq, verbose=1)

        np.save(os.path.join(self.cache_dir, 'debug', 'raw_predictions'), whales)

        # get array showing for each class where started it's embeddings
        ids = self.embeddings['Id']
        last_id, starts = ids[0], [0]
        for ind, curr_id in enumerate(ids):
            if last_id != curr_id:
                starts.append(ind)
                last_id = curr_id
        starts.append(len(ids))

        # get 2D array showing mean dist between val embedding and embeddings of group
        embeddings = self.embeddings.drop(['Id'], axis=1)
        mean_dist = np.empty((whales.shape[0], len(starts) - 1), dtype=float)
        for i in range(1, len(starts)):
            group_embeddings = embeddings[starts[i - 1]:starts[i]]

            concat = np.concatenate((group_embeddings, whales), axis=0)
            prod = np.dot(concat, np.transpose(concat))
            sq_norms = np.reshape(np.diag(prod), (-1, 1))

            dist = sq_norms - 2.0 * prod + np.transpose(sq_norms)
            dist = np.maximum(dist, 0.0)
            dist = dist[group_embeddings.shape[0]:, :group_embeddings.shape[0]]

            mean_dist[:, i - 1] = np.mean(dist, axis=1)

        predictions = np.apply_along_axis(np.argpartition, 1, mean_dist, 4)
        self.predictions = pd.DataFrame(data=predictions[:, :5])
        self.predictions = pd.concat([pd.DataFrame(data=whales_data), self.predictions], axis=1)
        self.predictions.columns = ['Image'] + list(range(5))
        self.save_predictions(os.path.join(self.cache_dir, 'predictions.pkl'))

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename, by_name=True, skip_mismatch=True)

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

    def _read_csv(self, csv, write_mappings=False, mappings_filename=None):
        csv_data = pd.read_csv(csv)
        whales = np.sort(csv_data['Id'].unique())
        if mappings_filename is not None:
            mapping = np.load(mappings_filename).item()
        else:
            mapping = {}
            reverse_mapping = {}
            for i, w in enumerate(whales):
                mapping[w] = i
                reverse_mapping[i] = [w]
            if write_mappings:
                np.save(os.path.join(self.cache_dir, 'whales_to_idx_mapping'), mapping)
                np.save(os.path.join(self.cache_dir, 'idx_to_whales_mapping'), reverse_mapping)
        data = csv_data.replace({'Id': mapping})
        return data.values

    @staticmethod
    def draw_tsne(vectors):
        tsne = TSNE(n_components=3, verbose=1, n_iter=300, perplexity=50).fit_transform(vectors)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2])
        plt.show()

    def make_kaggle_csv(self, mapping_file):
        mapping = np.load(mapping_file).item()
        for k in mapping:
            mapping[k] = mapping[k][0]
        predictions = self.predictions.replace({0: mapping, 1: mapping, 2: mapping, 3: mapping, 4: mapping})
        print(predictions)
        predictions['Id'] = predictions[0].astype('str') + ' ' + predictions[1].astype('str') + ' ' + predictions[2].astype('str') + ' ' + predictions[3].astype('str') + ' ' + predictions[4].astype('str')
        predictions.to_csv(os.path.join(self.cache_dir, 'submission.csv'), index=False, columns=['Image', 'Id'])

    def make_csv(self, mapping_file):
        mapping = np.load(mapping_file).item()
        for k in mapping:
            mapping[k] = mapping[k][0]
        predictions = self.predictions.replace({0: mapping, 1: mapping, 2: mapping, 3: mapping, 4: mapping})
        predictions['Id'] = predictions[0]
        predictions.to_csv(os.path.join(self.cache_dir, 'prediction.csv'), index=False, columns=['Image', 'Id'])

    def write_config(self):
        config = {
            'model': self.model_name,
            'input_shape': self.input_shape,
            'embedding_size': self.embedding_size,
            'strategy': self.strategy,
            'cache_dir': self.cache_dir
        }
        with open(os.path.join(self.cache_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

    @staticmethod
    def restore_from_config(filename):
        with open(filename) as f:
            config = json.load(f)
            assert 'model' in config
            assert 'input_shape' in config
            assert 'embedding_size' in config
            assert 'strategy' in config
            assert 'cache_dir' in config

            model = Siamese(config['model'], config['strategy'], config['input_shape'], config['embedding_size'])

            shutil.rmtree(model.cache_dir)
            cd = config['cache_dir']
            model.cache_dir = cd

            if os.path.isfile(os.path.join(cd, 'final_weights.h5')):
                model.load_weights(os.path.join(cd, 'final_weights.h5'))
                print('loaded weights from %s' % os.path.join(cd, 'final_weights.h5'))
            elif os.path.isdir(os.path.join(cd, 'training')):
                model.load_weights(os.path.join(cd, 'training', os.listdir(os.path.join(cd, 'training'))[-1]))
                print('loaded weights from %s' % os.path.join(cd, 'training', os.listdir(os.path.join(cd, 'training'))[-1]))

            if os.path.isfile(os.path.join(cd, 'embeddings.pkl')):
                model.load_embeddings(os.path.join(cd, 'embeddings.pkl'))
                print('loaded embeddings from %s' % os.path.join(cd, 'embeddings.pkl'))
            if os.path.isfile(os.path.join(cd, 'predictions.pkl')):
                model.load_predictions(os.path.join(cd, 'predictions.pkl'))
                print('loaded predictions from %s' % os.path.join(cd, 'predictions.pkl'))

            return model






