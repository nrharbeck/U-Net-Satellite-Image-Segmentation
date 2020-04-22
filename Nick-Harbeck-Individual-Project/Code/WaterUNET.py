
from __future__ import division

import numpy as np
import keras
from keras.utils import Sequence
from keras.layers import concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, BatchNormalization

from keras import backend as K

import h5py
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
from keras.backend import binary_crossentropy

import datetime
import os
import random
import matplotlib.pyplot as plt

img_rows = 112
img_cols = 112

smooth = 1e-12

num_channels = 22
num_mask_channels = 2


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def get_unet0():
    inputs = keras.Input((img_rows, img_cols, num_channels))
    conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization()(crop9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv10 = Conv2D(num_mask_channels, (1, 1), activation='sigmoid')(conv9)

    model = keras.Model(input=inputs, output=conv10)

    return model

def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows-32, img_cols-32))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        X_batch[i] = X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        yb = y[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        y_batch[i] = yb[:, 16:16 + img_rows - 32, 16:16 + img_cols - 32]
    return np.transpose(X_batch, (0, 2, 3, 1)), np.transpose(y_batch, (0, 2, 3, 1))

class data_generator(Sequence):

    def __init__(self, x_set, y_set, batch_size, horizontal_flip, vertical_flip, swap_axis):
        self.swap_axis = swap_axis
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        X_batch, y_batch = form_batch(self.x, self.y, self.batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    xb = np.fliplr(xb)
                    yb = np.fliplr(yb)

            if self.vertical_flip:
                if np.random.random() < 0.5:
                    xb = np.flipud(xb)
                    yb = np.flipud(yb)

            if self.swap_axis:
                if np.random.random() < 0.5:
                    xb = np.rot90(xb)
                    yb = np.rot90(yb)

            X_batch[i] = xb
            y_batch[i] = yb
        return X_batch, y_batch #Changed this from yield to return for running the same file

if __name__ == '__main__':
    data_path = os.getcwd()
    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet0()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    f = h5py.File(os.path.join(data_path, 'train_water.h5'), 'r')

    X_train = f['train']

    y_train = np.array(f['train_mask'])
    print(y_train.shape)

    train_ids = np.array(f['train_ids'])

    batch_size = 128
    nb_epoch = 4 

    filepath = "water.h5"
    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    #model.load_weights('water.h5') #Comment this if you have not already run the model at least once, it helps to save time in subsequent training steps.
    history = model.fit_generator(generator=data_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
                    epochs=nb_epoch,
                    verbose=1,
                    samples_per_epoch=batch_size * 100,
                    validation_data=data_generator(X_train, y_train, 128, horizontal_flip=False, vertical_flip=False, swap_axis=False),
                    validation_steps = 4,
                    callbacks=[ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, save_weights_only=True)],
                    workers=8
                    )

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['binary_crossentropy'])
    plt.plot(history.history['val_binary_crossentropy'])
    plt.title('model binary_crossentropy')
    plt.ylabel('binary_crossentropy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('water_binary_crossentropy' +str(history.history['val_jaccard_coef_int'][-1]) +'.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('water_loss' +str(history.history['val_jaccard_coef_int'][-1]) +'.png')

    f.close()

print(history.history.keys())