from __future__ import division

import os
from tqdm import tqdm
import pandas as pd
import extra_functions
import shapely.geometry
from numba import jit

from keras.models import model_from_json
import numpy as np


def read_model(cross=''):
    json_name = 'architecture_128_50_buildings_3_' + cross + '.json'
    weight_name = 'model_weights_128_50_buildings_3_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../cache', json_name)).read())
    model.load_weights(os.path.join('../cache', weight_name))
    return model

model = read_model()

sample = pd.read_csv('../sample_submission.csv')

data_path = os.getcwd()
num_channels = 24
num_mask_channels = 1
threashold = 0.3

three_band_path = os.path.join(data_path, 'three_band')

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

test_ids = shapes.loc[~shapes['image_id'].isin(train_wkt['ImageId'].unique()), 'image_id']

result = []

@jit
def mask2poly(predicted_mask, threashold, x_scaler, y_scaler):
    polygons = extra_functions.mask2polygons_layer(predicted_mask[0] > threashold, epsilon=0, min_area=5)

    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
    return shapely.wkt.dumps(polygons.buffer(2.6e-5))


for image_id in tqdm(test_ids):
    image = extra_functions.read_image_24(image_id)

    H = image.shape[1]
    W = image.shape[2]

    x_max, y_min = extra_functions._get_xmax_ymin(image_id)

    predicted_mask = extra_functions.make_prediction_cropped(model, image, initial_size=(112, 112),
                                                             final_size=(112-32, 112-32),
                                                             num_masks=num_mask_channels, num_channels=num_channels)

    image_v = np.flipud(image)
    predicted_mask_v = extra_functions.make_prediction_cropped(model, image_v, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    image_h = np.fliplr(image)
    predicted_mask_h = extra_functions.make_prediction_cropped(model, image_h, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    image_s = np.rot90(image)
    predicted_mask_s = extra_functions.make_prediction_cropped(model, image_s, initial_size=(112, 112),
                                                               final_size=(112 - 32, 112 - 32),
                                                               num_masks=1,
                                                               num_channels=num_channels)

    new_mask = np.power(predicted_mask *
                        np.flipud(predicted_mask_v) *
                        np.fliplr(predicted_mask_h) *
                        np.rot90(predicted_mask_s, 3), 0.25)

    x_scaler, y_scaler = extra_functions.get_scalers(H, W, x_max, y_min)

    mask_channel = 0
    result += [(image_id, mask_channel + 1, mask2poly(new_mask, threashold, x_scaler, y_scaler))]

submission = pd.DataFrame(result, columns=['ImageId', 'ClassType', 'MultipolygonWKT'])


sample = sample.drop('MultipolygonWKT', 1)
submission = sample.merge(submission, on=['ImageId', 'ClassType'], how='left').fillna('MULTIPOLYGON EMPTY')

submission.to_csv('temp_building_3.csv', index=False)
