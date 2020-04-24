#This dataloading is set up for the images from the DSTL dataset
import os
import numpy as np
import pandas as pd
#Guide to download Kaggle datasets directly found here: https://gist.github.com/jayspeidell/d10b84b8d3da52df723beacc5b15cb27
import kaggle
api_token = {"username":"USERNAME_GOES_HERE","key":"KEY_GOES_HERE"}
import json
import zipfile
import os
with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)
os.system('kaggle competitions download -c dstl-satellite-imagery-feature-detection')
if not os.path.exists("/content/competitions/dstl-satellite-imagery-feature-detection"):
    os.makedirs("/content/competitions/dstl-satellite-imagery-feature-detection")
os.chdir('/content/competitions/dstl-satellite-imagery-feature-detection')
for file in os.listdir():
  zip_ref = zipfile.ZipFile(file, 'r')
  zip_ref.close()
  os.system("unzip sixteen_band.zip")
  os.system("unzip grid_sizes.csv.zip")
  os.system("unzip sample_submissions.csv.zip")
  os.system("unzip three_band.zip")
  os.system("unzip train_geojson_v3.zip")
  os.system("unzip train_wkt_v4.zip")