from __future__ import division

import os
from tqdm import tqdm
import pandas as pd
import extra_functions
import tensorflow.keras
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, BatchNormalization
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import binary_crossentropy
import h5py
import numpy as np
from sklearn.metrics import jaccard_score

img_rows = 112
img_cols = 112
smooth = 1e-12

data_path = os.getcwd()
num_channels = 22
num_mask_channels = 2

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

def read_model(file):
    model = get_unet0()
    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    model.load_weights(file)
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

def calc_jacc(model):
    img, msk = form_batch(X_train, y_train, 128)

    prd = model.predict(img)
    print(prd.shape, msk.shape)

    for i in range(2):
        t_msk = msk[:, :, :, i]
        t_prd = prd[:, :, :, i]

        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10.0
            pred_binary_mask = (t_prd > tr) * 1.0

            jk = jaccard_coef(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print(i, m, b_tr)

f = h5py.File(os.path.join(data_path, 'train_water.h5'), 'r')

X_train = f['train']

y_train = np.array(f['train_mask'])
print(y_train.shape)

train_ids = np.array(f['train_ids'])
model = read_model('water.h5')
for i in range(11):
    calc_jacc(model)