import numpy as np
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import openslide
from openslide import deepzoom

slide_dir = 'D:/Dataset/Open-Sourced/Camelyon16/Tumor' # Directory storing WSIs
mask_dir = 'E:/Camelyon16 Segmentation/Training/Slide-level Segmentation Maps' # Directory storing normal-tumor segmentation maps
bg_tissue_dir = 'E:/Camelyon16 Segmentation/Training/Tissue-BG Segmentation Maps' # Directory storing the tissue-background binary maps
save_dir = 'E:/Camelyon16 Segmentation/Training/Training Set' # Directory to save the extracted tiles and their corresponding binary masks

#%% Extract relevant tiles from tumor-containing slides

case_list = os.listdir(mask_dir) 
case_list = [i.replace('_tumor_mask.tif','') for i in case_list]

tile_dim = 512
tile_overlap = 0
bg_threshold = 0.5

for case in tqdm(range(0,len(case_list))):
    current_case = case_list[case]
    print('Processing:',current_case)
    
    slide = openslide.OpenSlide(f'{slide_dir}/{current_case}.tif')
    mask = openslide.OpenSlide(f'{mask_dir}/{current_case}_tumor_mask.tif')
    bg_tissue_map = plt.imread(f'{bg_tissue_dir}/{current_case}.png')
    
    tile_save_dir = f'{save_dir}/Tissue'
    mask_save_dir = f'{save_dir}/Mask'
    
    # Create directiories for storing tissue tiles and mask tiles
    if not os.path.exists(tile_save_dir):
        os.mkdir(tile_save_dir)
    if not os.path.exists(mask_save_dir):
        os.mkdir(mask_save_dir)
    
    # Set Deepzoom generator for generating tiles
    tile_gen = deepzoom.DeepZoomGenerator(slide,tile_size=tile_dim,overlap=tile_overlap,limit_bounds=False) # Set tile generator for tissues
    tile_num = tile_gen.level_tiles[-1] # at highest magnification
    tile_hor_num = tile_num[0]-1 # Number of tiles along horizontal direction
    tile_ver_num = tile_num[1]-1 # Number of tiles along vertical direction
    
    mask_gen = deepzoom.DeepZoomGenerator(mask,tile_size=tile_dim,overlap=tile_overlap,limit_bounds=False) # Set tile generator for masks
    
    # Read tissue-bg map
    slide_dim = slide.dimensions # (horizontal pixel count, vertical pixel count)
    bg_tissue_map_dim = bg_tissue_map.shape # (vertical pixel count, horizontal pixel count)
    ratio = int(tile_dim / (int(slide_dim[0] / bg_tissue_map_dim[1]))) # Ratio for downscaling the tissue-bg map to become tile-level labeling for the tiles 
    new_bg_tissue_map = np.zeros((int(bg_tissue_map_dim[0] / ratio),int(bg_tissue_map_dim[1] / ratio))) # Create new background-tissue map
    
    # Downsample the original map to new map
    for y in range(0,new_bg_tissue_map.shape[0]):
        for x in range(0,new_bg_tissue_map.shape[1]):
            sub_region = bg_tissue_map[y*ratio:(y+1)*ratio,x*ratio:(x+1)*ratio]
            new_bg_tissue_map[y,x] = np.sum(sub_region) / (ratio * ratio)
    
    #plt.imshow(bg_tissue_map,cmap='gray')
    #plt.show()
    #plt.imshow(new_bg_tissue_map,cmap='gray')
    #plt.show()
    
    # Extract tiles
    for j in tqdm(range(0,tile_ver_num)):
        for i in range(0,tile_hor_num):
            if new_bg_tissue_map[j,i] >= bg_threshold: # Tiles with less than 50% of background are accepted
                mask_img = mask_img = np.array(mask_gen.get_tile(tile_gen.level_count-1,(i,j)))
                mask_img = mask_img[:,:,0]
                if 0 < np.sum(mask_img) < tile_dim * tile_dim: # Tiles containing the boundaries between normal tissues and tumor
                    tile_img = np.array(tile_gen.get_tile(tile_gen.level_count-1,(i,j)))
                    
                    # If the tiles look pale, then it may be not tissue-containing tiles, tile score is calculated according to overall brightness
                    tile_score = np.sum(tile_img,axis=2)
                    tile_score[tile_score<220*3] = 1
                    tile_score[tile_score>=220*3] = 0
                    tile_score = np.sum(tile_score)
                    
                    if tile_score >= tile_dim * tile_dim / 2: # If the tile is verified by tissue-background and overall color distribution is darker, it is very likely to be tissue-containing
                        # Save tiles
                        #np.save(f'{tile_save_dir}/{current_case}_{i}_{j}.npy',tile_img.astype('float32'))
                        plt.imsave(f'{tile_save_dir}/{current_case}_{i}_{j}.png',tile_img)
                        
                        # Save masks
                        np.save(f'{mask_save_dir}/{current_case}_{i}_{j}.npy',mask_img.astype('float32'))
                        #plt.imsave(f'{mask_save_dir}/{current_case}_{i}_{j}.png',mask_img,cmap='gray',vmin=0,vmax=1)

#%% Concatenate the extracted tiles and masks into training set

import os
import numpy as np
import random
import staintools
from tqdm import tqdm

random.seed(1)

datatype = 'float16'
sample = 5000
train = 4000
test = 1000

# Define stain normalization operation
def stain_norm(stain_ref,image):
    target = staintools.LuminosityStandardizer.standardize(stain_ref)
    to_transform = staintools.LuminosityStandardizer.standardize(image)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    transformed = normalizer.transform(to_transform)
    return transformed
    
tissue_dir = f'{save_dir}/Tissue'
mask_dir = f'{save_dir}/Mask'

tissue_list = os.listdir(tissue_dir)
tissue_list_sampled = random.sample(tissue_list,sample) # Randomly select 5000 samples as training and testing set

stain_ref = staintools.read_image('E:/Camelyon16 Segmentation/Training/Training Set/Reference Image.jpg')

x_train = np.zeros((train,512,512,3),dtype=datatype)
x_test = np.zeros((test,512,512,3),dtype=datatype)
y_train = np.zeros((train,512,512,1),dtype=datatype)
y_test = np.zeros((test,512,512,1),dtype=datatype)

for i in tqdm(range(0,len(tissue_list_sampled))):
    # Load tissue tiles and stain normalization 
    tissue_img = staintools.read_image(f'{tissue_dir}/{tissue_list_sampled[i]}')
    tissue_img_sn = stain_norm(stain_ref,tissue_img)
    
    # Load mask tiles 
    mask_img = np.load(f'{mask_dir}/{tissue_list_sampled[i].replace("png","npy")}')
    
    if i < train: # First 4000 --> training set
        x_train[i,:,:,:] = tissue_img_sn
        y_train[i,:,:,:] = mask_img
        
    else: # Next 1000 --> testing set
        x_test[i-train,:,:,:] = tissue_img_sn
        y_test[i-train,:,:,:] = mask_img

# Save training set and testing set
np.save(f'{save_dir}/x_train.npy',x_train)
np.save(f'{save_dir}/x_test.npy',x_test)
np.save(f'{save_dir}/y_train.npy',y_train)
np.save(f'{save_dir}/y_test.npy',y_test)
