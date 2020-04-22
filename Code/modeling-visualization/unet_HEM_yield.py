from __future__ import division

import numpy as np
import tensorflow.keras
from tensorflow.keras.utils import Sequence
import threading
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, BatchNormalization
from tensorflow.keras import backend as K

import h5py
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy

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
    inputs = Input((img_rows, img_cols, num_channels))
    conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = tensorflow.keras.layers.ELU()(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = tensorflow.keras.layers.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = tensorflow.keras.layers.ELU()(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = tensorflow.keras.layers.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = tensorflow.keras.layers.ELU()(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = tensorflow.keras.layers.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = tensorflow.keras.layers.ELU()(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = tensorflow.keras.layers.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = tensorflow.keras.layers.ELU()(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = tensorflow.keras.layers.ELU()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = tensorflow.keras.layers.ELU()(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = tensorflow.keras.layers.ELU()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = tensorflow.keras.layers.ELU()(conv7)
    conv7 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = tensorflow.keras.layers.ELU()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = tensorflow.keras.layers.ELU()(conv8)
    conv8 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = tensorflow.keras.layers.ELU()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = tensorflow.keras.layers.ELU()(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization()(crop9)
    conv9 = tensorflow.keras.layers.ELU()(conv9)
    conv10 = Conv2D(num_mask_channels, (1, 1), activation='sigmoid')(conv9)

    model = tensorflow.keras.Model(inputs=inputs, outputs=conv10)

    return model

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

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

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def mine_hard_samples(model, datagen, batch_size):
    while True:
        samples, targets, loss = [], [], []
        x_data, y_data = next(datagen)
        preds = model.predict(x_data)
        for i in range(len(preds)):
            loss.append(K.mean(jaccard_coef_loss(y_data[i], preds[i])))
        ind = np.argpartition(np.asarray(loss), -int(batch_size / 2))[-int(batch_size / 2):]
        samples += x_data[ind].tolist()
        targets += y_data[ind].tolist()

        x_data, y_data = next(datagen)
        samples += x_data[:int(batch_size/2)].tolist()
        targets += y_data[:int(batch_size/2)].tolist()
        samples, targets = map(np.array, (samples, targets))

        for i in range(batch_size):
            xb = samples[i]
            yb = targets[i]

            if np.random.random() < 0.5:
                xb = np.fliplr(xb)
                yb = np.fliplr(yb)

            if np.random.random() < 0.5:
                xb = np.flipud(xb)
                yb = np.flipud(yb)

            if np.random.random() < 0.5:
                xb = np.rot90(xb)
                yb = np.rot90(yb)

            samples[i] = xb
            targets[i] = yb

        yield samples, targets

@threadsafe_generator
def gen(batch_size):
    while True:
        x_data, y_data = form_batch(X_train, y_train, batch_size)
        yield x_data, y_data

if __name__ == '__main__':
    data_path = os.getcwd()
    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet0()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    f = h5py.File(os.path.join(data_path, 'train_b_s.h5'), 'r')

    X_train = f['train']

    y_train = np.array(f['train_mask'])
    print(y_train.shape)

    train_ids = np.array(f['train_ids'])

    batch_size = 128
    nb_epoch = 5

    filepath = "b_s.h5"
    model.compile(optimizer=Nadam(lr=1e-4), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    model.load_weights('b_s.h5')
    x, y = next(gen(batch_size))
    model.predict(x)
    history = model.fit_generator(generator=mine_hard_samples(model, gen(batch_size), batch_size),
                                  epochs=nb_epoch,
                                  verbose=1,
                                  steps_per_epoch=40,
                                  validation_data=gen(batch_size),
                                  validation_steps=4,
                                  callbacks=[ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, save_weights_only=True)],
                                  workers=8
                                  )

    plt.plot(history.history['binary_crossentropy'])
    plt.plot(history.history['val_binary_crossentropy'])
    plt.title('model binary_crossentropy')
    plt.ylabel('binary_crossentropy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('b_s_binary_crossentropy' +str(np.min(history.history['val_binary_crossentropy'])) +'.png')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('b_s_loss' +str(np.max(history.history['val_jaccard_coef_int'])) +'.png')

    f.close()