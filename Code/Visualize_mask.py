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
#rgb = tiff.imread(data_path + '/three_band/6110_4_0.tif')
#rgb = np.rollaxis(rgb, 0, 3)
#cv2.imwrite('org.png',255 * stretch_8bit(rgb))

f = h5py.File(os.path.join(data_path, 'train_test.h5'), 'r')

X_train = f['train_mask'][0]
#print(f['train_ids'])
img = np.transpose(X_train, (1, 2, 0))
#img = img[:,:, 21:]

img = np.concatenate([img, np.expand_dims(img[:, :, 0], 2)], axis=2)
img = 255 * img
img = img.astype(np.uint8)
cv2.imwrite('mask.png',img)
