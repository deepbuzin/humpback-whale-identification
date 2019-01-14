from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPool2D, ZeroPadding2D
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.utils import plot_model

import cv2
import numpy as np


def identity(x, filters, kernel_size, stage, block, regularizer=None, trainable=True):
    f1, f2, f3 = filters
    shortcut = x

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv%d_%d_a' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_a' % (stage, block))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1), padding='same', name='conv%d_%d_b' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_b' % (stage, block))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv%d_%d_c' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_c' % (stage, block))(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv(x, filters, kernel_size, stage, block, strides=(2, 2), regularizer=None, trainable=True):
    f1, f2, f3 = filters
    shortcut = x

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides, padding='valid', name='conv%d_%d_a' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_a' % (stage, block))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1), padding='same', name='conv%d_%d_b' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_b' % (stage, block))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv%d_%d_c' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_c' % (stage, block))(x)

    shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=strides, padding='valid', name='conv%d_%d_s' % (stage, block),
                      kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(shortcut)
    shortcut = BatchNormalization(axis=3, name='bn%d_%d_s' % (stage, block))(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


if __name__ == '__main__':
    input_shape = (224, 224, 3)
    num_classes = 1000

    img = Input(input_shape)
    x = ZeroPadding2D((3, 3))(img)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1', kernel_initializer=glorot_uniform())(x)
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
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=2)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=3)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=4)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=5)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=6)

    x = conv(x, filters=[512, 512, 2048], kernel_size=(3, 3), stage=5, block=1)
    x = identity(x, filters=[512, 512, 2048], kernel_size=(3, 3), stage=5, block=2)
    x = identity(x, filters=[512, 512, 2048], kernel_size=(3, 3), stage=5, block=3)

    x = AveragePooling2D((7, 7))(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc%d' % num_classes, kernel_initializer=glorot_uniform())(x)

    model = Model(inputs=img, outputs=x, name='ResNet50')
    model.summary()

    model.load_weights('imagenet.h5')

    cat = cv2.imread('cat.jpg')
    cat = cv2.cvtColor(cat, cv2.COLOR_BGR2RGB)
    print(np.argmax(model.predict(np.asarray([cat]))))

