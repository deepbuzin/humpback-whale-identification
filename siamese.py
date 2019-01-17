from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

from resnet.resnet import identity, conv
from triplet_loss.triplet_loss import triplet_loss_batch_hard


def build():
    img = Input(shape=(512, 384, 3))

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

    x = AveragePooling2D((8, 6))(x)
    x = Flatten()(x)
    x = Dense(192, kernel_initializer=glorot_uniform())(x)

    model = Model(inputs=img, outputs=x, name='ResNet_siamese')
    model.summary()
    return model


if __name__ == '__main__':
    m = build()
    m.compile(optimizer=Adam(0.001), loss=triplet_loss_batch_hard(margin=0.2))


