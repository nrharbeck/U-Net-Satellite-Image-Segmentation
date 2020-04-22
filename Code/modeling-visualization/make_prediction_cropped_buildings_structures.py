"""
This code can be modified to visualize predicts of any model.
"""

from __future__ import division

import os
from tqdm import tqdm
import pandas as pd
import extra_functions
import tensorflow.keras
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import binary_crossentropy
import shapely.geometry
from numba import jit
import numpy as np

img_rows = 112
img_cols = 112
smooth = 1e-12

data_path = os.getcwd()
num_channels = 22
num_mask_channels = 2
threashold = 0.3

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

model = read_model('b_s.h5')

sample = pd.read_csv('sample_submission.csv')

three_band_path = os.path.join(data_path, 'three_band')

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

#test_ids = shapes.loc[~shapes['image_id'].isin(train_wkt['ImageId'].unique()), 'image_id']
test_ids = ['6050_4_4', '6060_0_1', '6060_1_4', '6100_0_2', '6100_2_4', '6110_2_3', '6120_1_4', '6120_3_3']
result = []

@jit
def mask2poly(predicted_mask, threashold, x_scaler, y_scaler):
    polygons = extra_functions.mask2polygons_layer(predicted_mask > threashold, epsilon=0, min_area=5)

    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
    return shapely.wkt.dumps(polygons.buffer(2.6e-5))


#for image_id in tqdm(test_ids[:2]):
for image_id in test_ids:
    print(image_id)
    image = extra_functions.read_image_22(image_id)

    H = image.shape[0]
    W = image.shape[1]

    x_max, y_min = extra_functions._get_xmax_ymin(image_id)

    predicted_mask = extra_functions.make_prediction_cropped(model, image, initial_size=(112, 112),
                                                             final_size=(112-32, 112-32),
                                                             num_masks=num_mask_channels, num_channels=num_channels)

    image_v = np.flipud(image)
    predicted_mask_v = extra_functions.make_prediction_cropped(model, image_v, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=2,
                                                               num_channels=num_channels)

    image_h = np.fliplr(image)
    predicted_mask_h = extra_functions.make_prediction_cropped(model, image_h, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=2,
                                                               num_channels=num_channels)

    image_s = np.rot90(image)
    predicted_mask_s = extra_functions.make_prediction_cropped(model, image_s, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=2,
                                                               num_channels=num_channels)

    new_mask = np.power(predicted_mask *
                        np.flipud(predicted_mask_v) *
                        np.fliplr(predicted_mask_h) *
                        np.rot90(predicted_mask_s, 3), 0.25)

    x_scaler, y_scaler = extra_functions.get_scalers(H, W, x_max, y_min)

    mask_channel = 0
    result += [(image_id, mask_channel + 1, mask2poly(new_mask[:, :, 0], threashold, x_scaler, y_scaler))]
    mask_channel = 1
    result += [(image_id, mask_channel + 1, mask2poly(new_mask[:, :, 1], threashold, x_scaler, y_scaler))]

submission = pd.DataFrame(result, columns=['ImageId', 'ClassType', 'MultipolygonWKT'])


sample = sample.drop('MultipolygonWKT', 1)
submission = sample.merge(submission, on=['ImageId', 'ClassType'], how='left').fillna('MULTIPOLYGON EMPTY')

submission.to_csv('temp_b_s.csv', index=False)