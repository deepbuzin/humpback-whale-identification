from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from keras.models import Model, load_model
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

from resnet.resnet import identity, conv
from triplet_loss.triplet_loss import triplet_loss_batch_hard
from utils.sequence import WhalesSequence


def build_modest():
    img = Input(shape=(384, 512, 3))

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1', kernel_initializer=glorot_uniform())(img)
    x = BatchNormalization(axis=3, name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv(x, filters=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), stage=2, block=1)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=2)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=3)

    x = conv(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=1)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=2)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=3)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=4)

    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=5, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=6, block=1)

    x = AveragePooling2D((6, 8))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer=glorot_uniform())(x)
    x = Dense(192, kernel_initializer=glorot_uniform(), kernel_regularizer='l2')(x)

    model = Model(inputs=img, outputs=x, name='ResNet_siamese')
    model.summary()
    return model


def build():
    img = Input(shape=(768, 1024, 3))

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1', kernel_initializer=glorot_uniform())(img)
    x = BatchNormalization(axis=3, name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv(x, filters=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), stage=2, block=1)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=2)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=3)

    x = conv(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=1)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=2)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=3)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=4)

    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=5, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=6, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=7, block=1)

    x = AveragePooling2D((6, 8))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer=glorot_uniform())(x)
    x = Dense(192, kernel_initializer=glorot_uniform(), kernel_regularizer='l2')(x)

    model = Model(inputs=img, outputs=x, name='ResNet_siamese')
    model.summary()
    return model


def fit(model, img_dir, csv):
    data = np.genfromtxt(csv, dtype=str, delimiter=',', skip_header=True)
    train = WhalesSequence(img_dir, input_shape=(384, 512, 3), x_set=data[:, 0], y_set=data[:, 1], batch_size=10)
    model.fit_generator(train, epochs=2)
    model.save('model.h5')


def predict(model, img_dir):
    data = np.array(os.listdir(img_dir))
    train = WhalesSequence(img_dir, input_shape=(384, 512, 3), x_set=data, batch_size=10)
    emb = model.predict_generator(train, verbose=1)
    print(emb)
    np.save('np_embeddings.pkl', emb)


if __name__ == '__main__':
    # m = build_modest()
    # m.compile(optimizer=Adam(0.001), loss=triplet_loss_batch_hard(margin=0.2))
    # fit(m, 'D:/IdeaProjects/whales/data/train', 'D:/IdeaProjects/whales/data/train_fixed.csv')

    m_trained = load_model('model.h5', custom_objects={'batch_hard': triplet_loss_batch_hard(margin=0.2)})
    predict(m_trained, 'D:/IdeaProjects/whales/data/train')



