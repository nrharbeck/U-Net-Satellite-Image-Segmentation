"""
Script that caches train data for future training
"""

from __future__ import division

import os
import pandas as pd
import extra_functions
import h5py
import numpy as np
import cv2


data_path = os.getcwd()

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

def cache_train_t_c():

    image_set = extra_functions.get_class_image(classes=[5, 6])

    num_train = len(image_set)

    print('num_train_images =', num_train)

    train_shapes = shapes[shapes['image_id'].isin(image_set)]

    image_rows = train_shapes['height'].min()
    image_cols = train_shapes['width'].min()

    num_channels = 24

    num_mask_channels = 2

    f = h5py.File(os.path.join(data_path, 'train_t_c.h5'), 'w')

    imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16, compression='gzip', compression_opts=9)
    imgs_mask = f.create_dataset('train_mask', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8, compression='gzip', compression_opts=9)

    ids = []

    i = 0
    for image_id in image_set:
        image = extra_functions.read_image_24(image_id)
        height, width, _ = image.shape

        imgs[i] = np.transpose(cv2.resize(image, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC), (2, 0, 1))
        imgs_mask[i] = np.transpose(
            cv2.resize(np.transpose(extra_functions.generate_mask(image_id, height, width, start=4,
                                                                  num_mask_channels=num_mask_channels,
                                                                  train=train_wkt), (1, 2, 0)),
                       (image_cols, image_rows), interpolation=cv2.INTER_CUBIC), (2, 0, 1))

        ids += [image_id]
        i += 1

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')

    f.close()


if __name__ == '__main__':
    cache_train_t_c()
