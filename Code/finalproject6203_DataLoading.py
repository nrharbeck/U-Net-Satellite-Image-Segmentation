#Some image processing taken from Github guide: https://github.com/amir-jafari/Deep-Learning/blob/master/Exam_MiniProjects/5-Keras_Exam1_Sample_Codes_S20/load_data.py 
#This dataloading is set up for the three band images from the dataset
import os
import numpy as np
import pandas as pd

import cv2
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torchvision import datasets
import matplotlib.pyplot as plt 

#Guide to download Kaggle datasets directly found here: https://gist.github.com/jayspeidell/d10b84b8d3da52df723beacc5b15cb27
!pip install kaggle
#Insert your username and Kaggle API here
api_token = {"username":"USERNAME","key":"API_KEY"}
import json
import zipfile
import os
!pip install tifffile
import tifffile as tiff

if "three_band" not in os.listdir():
  with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)
  !kaggle datasets download -d dstl-satellite-imagery-feature-detection
  if not os.path.exists("/content/competitions/dstl-satellite-imagery-feature-detection"):
    os.makedirs("/content/competitions/dstl-satellite-imagery-feature-detection")
  os.chdir('/content/competitions/dstl-satellite-imagery-feature-detection')
  for file in os.listdir():
    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.close()
    os.system("unzip grid_sizes.csv.zip")
    os.system("unzip sample_submissions.csv.zip")
    os.system("unzip three_band.zip")
    os.system("unzip train_geojson_v3.zip")
    os.system("unzip train_wkt_v4.zip")
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

X_DATA_DIR = os.getcwd() + "/three_band/"
Y_DATA_DIR = os.getcwd() + "/train/geojson_v3/"

#I think it might be good to keep the color bands separate so we have more examples, it should act as an augmentation of our data so the model recognizes the key patterns of each feature we're trying to detect
x= []
for path in [f for f in os.listdir(X_DATA_DIR) if f[-4:] == ".tif"]:
    x.append(tiff.imread(X_DATA_DIR + path))
x = np.array(x)