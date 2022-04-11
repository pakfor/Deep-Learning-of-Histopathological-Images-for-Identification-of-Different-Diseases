import pandas as pd
import os 
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import staintools

def stain_norm(stain_ref,image):
    target = staintools.LuminosityStandardizer.standardize(stain_ref)
    to_transform = staintools.LuminosityStandardizer.standardize(image)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    transformed = normalizer.transform(to_transform)
    return transformed

ref_img = staintools.read_image('E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Tiles/Stain Normalization Reference.png')

gene_tile = pd.read_csv('E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Tiles/tumor_labels.txt',delimiter=' ',header=None,names=['gene','slide'])
save_selected_tile = 'E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Tiles/Tumor (3000)/Original'
original_path = 'E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Tiles/Tumor (All)'
gene_list = ['EGFR','KRAS','STK11']

# Selection: 3000 / gene 
for gene_i in tqdm(range(1,len(gene_list))):
    cur_consider_gene = gene_list[gene_i]
    rev_slide = gene_tile[gene_tile['gene'] == cur_consider_gene] 
    rev_slide = rev_slide.reset_index(drop=True)
    num_rev_slide = len(rev_slide)
    num_tile_per_slide = 3000 // num_rev_slide # Sample equal number of tiles from each slide
    num_extra_tile = 3000 - num_tile_per_slide*num_rev_slide # The remainder will be collected from the last slide in the list
    for slide_i in tqdm(range(0,num_rev_slide)):
        selected_slide = f'{original_path}/{rev_slide["slide"][slide_i]}'
        tile_in_selected_slide = os.listdir(selected_slide)
        
        if len(tile_in_selected_slide) < num_tile_per_slide: # If the tiles available in a slide is less than the designated amount, extract all tiles and left the remainder to the last slide
            num_to_extract = len(tile_in_selected_slide)
            num_extra_tile += num_tile_per_slide - len(tile_in_selected_slide)
            
        else: # len() < num_tile
            num_to_extract = num_tile_per_slide

        if slide_i != num_rev_slide - 1:
            selected_tile = random.sample(tile_in_selected_slide,num_to_extract) # Randomly sample the tiles within the same slide
        else:
            selected_tile = random.sample(tile_in_selected_slide,num_to_extract + num_extra_tile) # Sample more for the last slide in the list
        
        num_selected_tile = len(selected_tile)
        for tile_i in range(0,num_selected_tile):
            tile_load = staintools.read_image(f'{selected_slide}/{selected_tile[tile_i]}')
            tile_load = stain_norm(tile_load,ref_img)
            
            if tile_load.shape == (512,512,3): # Save only if the tile is 512 x 512 x 3
                plt.imsave(f'{save_selected_tile}/{cur_consider_gene}/{rev_slide["slide"][slide_i]}_{selected_tile[tile_i]}',tile_load)

#%% Concatenate the images into numpy arrays

import numpy as np
import os 

datatype = 'float16'
allTile = 'E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Tiles/Tumor (3000)/Stain Normalized'
allEGFR = os.listdir(f'{allTile}/EGFR') # Read tiles stored in EGFR folder
allKRAS = os.listdir(f'{allTile}/KRAS') # Read tiles stored in KRAS folder
allSTK = os.listdir(f'{allTile}/STK11') # Read tiles stored in STK11 folder

# Create arrays that store the tiles from individual gene mutations
EGFR = np.zeros((len(allEGFR),512,512,3),dtype=datatype) 
KRAS = np.zeros((len(allKRAS),512,512,3),dtype=datatype)
STK11 = np.zeros((len(allSTK),512,512,3),dtype=datatype)

# Create array that store the labels to the tiles
y_test = np.zeros((len(allEGFR) + len(allKRAS) + len(allSTK),1),dtype=datatype)

for i in range(0,len(allEGFR)):
    EGFR[i,:,:,:] = plt.imread(f'{allTile}/EGFR/{allEGFR[i]}')
    y_test[i,:] = 0
    
for i in range(0,len(allKRAS)):
    KRAS[i,:,:,:] = plt.imread(f'{allTile}/KRAS/{allKRAS[i]}')
    y_test[i + len(allEGFR),:] = 1
    
for i in range(0,len(allSTK)):
    STK11[i,:,:,:] = plt.imread(f'{allTile}/STK11/{allSTK[i]}')
    y_test[i + len(allEGFR) + len(allKRAS),:] = 2

# Concatenate the numpy arrays
x_test = np.concatenate((EGFR,KRAS,STK11),axis=0)
x_test = x_test.astype(datatype)

np.save('E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Array/x_test.npy',x_test)
np.save('E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Array/y_test.npy',y_test)