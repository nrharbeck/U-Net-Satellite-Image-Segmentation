# Final-Project-Group2
DATS 6203 Final Project 
This project uses a U-Net to classify and segment features from the 2016 United Kingdom’s Defence Science and Technology Laboratory (DSTL) Kaggle Competition, which can be found here: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection

Each folder in this repository contains various types of information. There is an additional README for the code folder for anyone that would like to test it out.

## Prepare data for training
1. Run python get_3_band_shapes.py to store the shape of all the images.
2. Run cache_train_b_s.py, cache_train_r_t, cache_train_t_c, cache_train_water, and cache_train_vehicle to store the images used for training each similar group in h5py file.
## Train models
Each similar group was trained and predicted using seperate U-Net.
1. For building and structures classes. Run python unet_buildings_structures. After model trained on certain epochs, run unet_b_s_HEM, which contains function of hard example mining.
2. Same for other groups, except water.
## Prediction
1. Run get_threshold.py to get the threshold for predicting each class
2. Run make_prediction_all_classes, which save prediction into pred_all.csv

