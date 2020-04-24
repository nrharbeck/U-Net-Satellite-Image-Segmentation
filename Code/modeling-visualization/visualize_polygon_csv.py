import os
import pandas as pd
import numpy as np
import cv2
import extra_functions

data_path = os.getcwd()
num_channels = 22
num_mask_channels = 2
pred = pd.read_csv('temp_b_s.csv')
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))
#test_id = pred['ImageId']
test_id = ['6050_4_4', '6060_0_1', '6060_1_4', '6100_0_2', '6100_2_4', '6110_2_3', '6120_1_4', '6120_3_3']

for image_id in test_id:
    print(image_id)
    mask = extra_functions.generate_mask(image_id, int(shapes[shapes['image_id'] == image_id]['height']),
                                         int(shapes[shapes['image_id'] == image_id]['width']), start=0,
                                         num_mask_channels=num_mask_channels, train=pred)
    mask = np.transpose(mask, (1, 2, 0))
    mask = extra_functions.stretch_n(mask)
    img = np.concatenate([mask, np.expand_dims(mask[:, :, 0], 2)], axis=2)
    img = 255 * img
    img = img.astype(np.uint8)
    cv2.imwrite('mask' + image_id +'.png', img)