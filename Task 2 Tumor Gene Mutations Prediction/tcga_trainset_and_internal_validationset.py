'''
TCGA-LUAD Training Set + Internal Validation Set Tile Extraction

1. Training + Testing Set (6000 / gene)
    - Training: 14400 x 512 x 512 x 3, FP16 (4800 / gene)
    - Testing: 3600 x 512 x 512 x 3, FP16 (1200 / gene)

2. Internal Validation Set (2000 / gene)
    - Validation: 6000 x 512 x 512 x 3, FP16 (2000 / gene)

Remarks:
    - Training, testing and validation set must not overlap
'''

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import staintools
import random

# Define stain normalization operation
def stain_norm(stain_ref,image):
    target = staintools.LuminosityStandardizer.standardize(stain_ref)
    to_transform = staintools.LuminosityStandardizer.standardize(image)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    transformed = normalizer.transform(to_transform)
    return transformed

# Define directories
allTile = 'E:/TCGA LUAD Gene Mutation Prediction/Data/TCGA LUAD Extracted Tiles (3CL Genes)'
allEGFR = f'{allTile}/EGFR'
allKRAS = f'{allTile}/KRAS'
allSTK = f'{allTile}/STK11'
save_dir = f'{allTile}/Training'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Basic variables
datatype = 'float16'
train = 4800
test = 1200
val = 2000
pergene = train + test + val

x_train = np.zeros((train*3,512,512,3),dtype=datatype)
y_train = np.zeros((train*3,1),dtype=datatype)
x_test = np.zeros((test*3,512,512,3),dtype=datatype)
y_test = np.zeros((test*3,1),dtype=datatype)
x_val = np.zeros((val*3,512,512,3),dtype=datatype)
y_val = np.zeros((val*3,1),dtype=datatype)

stain_ref = staintools.read_image('E:/TCGA LUAD Gene Mutation Prediction/Data/Stain Normalization Reference.png') # Reference image for stain normalization

# Random sample for each gene and concatenate them into numpy 
EGFR_list_sampled = random.sample(os.listdir(allEGFR),pergene)
KRAS_list_sampled = random.sample(os.listdir(allKRAS),pergene)
STK_list_sampled = random.sample(os.listdir(allSTK),pergene)

for i in tqdm(range(0,len(EGFR_list_sampled))):
    tile_img = staintools.read_image(f'{allEGFR}/{EGFR_list_sampled[i]}')
    tile_img_sn = stain_norm(stain_ref,tile_img) # Stain normalize the tile
    
    label = 0 # 0 for EGFR
    
    if i < train:
        x_train[i,:,:,:] = tile_img_sn
        y_train[i,:] = label
    
    if train < i < train + test:
        x_test[i - train,:,:,:] = tile_img_sn
        y_test[i - train,:] = label
    
    if i > train + test:
        x_val[i - train - test,:,:,:] = tile_img_sn
        y_val[i - train - test,:] = label

for i in tqdm(range(0,len(KRAS_list_sampled))):
    tile_img = staintools.read_image(f'{allKRAS}/{KRAS_list_sampled[i]}')
    tile_img_sn = stain_norm(stain_ref,tile_img) # Stain normalize the tile
    
    label = 1 # 1 for KRAS
    
    if i < train:
        x_train[i + train,:,:,:] = tile_img_sn
        y_train[i + train,:] = label
    
    if train < i < train + test:
        x_test[i - train + test,:,:,:] = tile_img_sn
        y_test[i - train + test,:] = label
    
    if i > train + test:
        x_val[i - train - test + val,:,:,:] = tile_img_sn
        y_val[i - train - test + val,:] = label
        
for i in tqdm(range(0,len(STK_list_sampled))):
    tile_img = staintools.read_image(f'{allSTK}/{STK_list_sampled[i]}')
    tile_img_sn = stain_norm(stain_ref,tile_img) # Stain normalize the tile
    
    label = 2 # 2 for STK11
    
    if i < train:
        x_train[i + 2 * train,:,:,:] = tile_img_sn
        y_train[i + 2 * train,:] = label
    
    if train < i < train + test:
        x_test[i - train + 2 * test,:,:,:] = tile_img_sn
        y_test[i - train + 2 * test,:] = label
    
    if i > train + test:
        x_val[i - train - test + 2 * val,:,:,:] = tile_img_sn
        y_val[i - train - test + 2 * val,:] = label

# Save all the numpy array
np.save(f'{save_dir}/x_train.npy',x_train)
np.save(f'{save_dir}/y_train.npy',y_train)
np.save(f'{save_dir}/x_test.npy',x_test)
np.save(f'{save_dir}/y_test.npy',y_test)
np.save(f'{save_dir}/x_val.npy',x_val)
np.save(f'{save_dir}/y_val.npy',y_val)