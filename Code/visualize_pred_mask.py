import os
import pandas as pd
import numpy as np
import cv2
import extra_functions
import shapely
from PIL import Image
import random
random.seed(0)
data_path = os.getcwd()
num_channels = 22
num_mask_channels = 10
pred = pd.read_csv('train_water13.csv')
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))
#test_id = pred['ImageId']
test_id = ['6070_2_3']
def polygons2mask_layer(height, width, polygon, image_id, i):
    """

    :param height:
    :param width:
    :param polygons:
    :return:
    """

    x_max, y_min = extra_functions._get_xmax_ymin(image_id)
    x_scaler, y_scaler = extra_functions.get_scalers(height, width, x_max, y_min)

    polygons = shapely.affinity.scale(polygon, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    img_mask = np.zeros((height, width), np.uint8)

    if not polygons:
        return img_mask

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 11-i)
    cv2.fillPoly(img_mask, interiors, 0)

    return img_mask


def generate_mask(image_id, height, width, train):
    """

    :param image_id:
    :param height:
    :param width:
    :param num_mask_channels: numbers of channels in the desired mask
    :param train: polygons with labels in the polygon format
    :return: mask corresponding to an image_id of the desired height and width with desired number of channels
    """

    mask = np.zeros((height, width, 10))
    for i in range(10):
        poly = train.loc[(train['ClassType'] == i+1), 'MultipolygonWKT'].values[0]

        polygons = shapely.wkt.loads(poly)

        mask[:, :, i] = polygons2mask_layer(height, width, polygons, image_id, i+1)

    return mask


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

for image_id in test_id:
    print(image_id)
    mask = generate_mask(image_id, int(shapes[shapes['image_id'] == image_id]['height']),
                                         int(shapes[shapes['image_id'] == image_id]['width']), train=pred)
    mask = np.argmax(mask, axis=2)
    palette = [random.randint(0, 225) for x in range(256 * 3)]  # 随机颜色的调色板

    mask_img = colorize_mask(mask)

    mask_img.save('pred_mask' + image_id +'.png')
