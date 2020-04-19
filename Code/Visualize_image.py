import os
import h5py
import cv2
import numpy as np
import tifffile as tiff
data_path = os.getcwd()

def stretch_8bit(bands, lower_percent=5, higher_percent=95):
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(3):
        a = 0
        b = 1
        c = np.percentile(bands[:,:, i], lower_percent)
        d = np.percentile(bands[:,:, i], higher_percent)
        t = a + (bands[:,:, i] - c) * (b - a) / (d - c)
        t[t<a] = a
        t[t>b] = b
        out[:,:, i] =t
    return out.astype(np.float32)

real_test_ids = ['6080_4_4', '6080_4_1', '6010_0_1', '6150_3_4', '6020_0_4', '6020_4_3',
                 '6150_4_3', '6070_3_4', '6020_1_3', '6060_1_4', '6050_4_4', '6110_2_3',
                 '6060_4_1', '6100_2_4', '6050_3_3', '6100_0_2', '6060_0_0', '6060_0_1',
                 '6060_0_3', '6060_2_0', '6120_1_4', '6160_1_4', '6120_3_3', '6140_2_3',
                 '6090_3_2', '6090_3_4', '6170_4_4', '6120_4_4', '6030_1_4', '6120_0_2',
                 '6030_1_2', '6160_0_0']
for i in real_test_ids:
    rgb = tiff.imread(data_path + '/three_band/' + i +'.tif')
    rgb = np.rollaxis(rgb, 0, 3)
    cv2.imwrite(i+'.png', 255 * stretch_8bit(rgb))

#f = h5py.File(os.path.join(data_path, 'train_test.h5'), 'r')

#X_train = f['train'][0]
#print(f['train_ids'])
#img = np.transpose(X_train, (1, 2, 0))
#img = img[:,:, 19:]
#img = 255 * img
#img = img.astype(np.float32)
#cv2.imwrite('rgb.png',img)
