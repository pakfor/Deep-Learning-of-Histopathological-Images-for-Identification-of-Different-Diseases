import numpy as np
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import openslide
from openslide import deepzoom

slide_dir = 'Camelyon16/Data/Test/Slides' # Directory storing WSIs
mask_dir = 'Camelyon16/Data/Test/Tumor Masks' # Directory storing normal-tumor segmentation maps
bg_tissue_dir = 'Camelyon16/Data/Test/Tissue-BG Masks' # Directory storing the tissue-background binary maps
save_dir = 'Camelyon16/Testing/Testing Set' # Directory to save the extracted tiles and their corresponding binary masks

#%% Extract cases that contain both normal tissues and tumor tissues in the slides

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
    
    case_save_dir = f'{save_dir}/{current_case}'
    tile_save_dir = f'{case_save_dir}/Tissue'
    mask_save_dir = f'{case_save_dir}/Mask'
    
    # Create directiories for storing tissue tiles and mask tiles
    if not os.path.exists(case_save_dir):
        os.mkdir(case_save_dir)
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
                tile_img = np.array(tile_gen.get_tile(tile_gen.level_count-1,(i,j)))
                
                # If the tiles look pale, then it may be not tissue-containing tiles, tile score is calculated according to overall brightness
                tile_score = np.sum(tile_img,axis=2)
                tile_score[tile_score<220*3] = 1
                tile_score[tile_score>=220*3] = 0
                tile_score = np.sum(tile_score)
                
                if tile_score >= tile_dim * tile_dim / 2: # If the tile is verified by tissue-background and overall color distribution is darker, it is very likely to be tissue-containing
                    mask_img = np.array(mask_gen.get_tile(tile_gen.level_count-1,(i,j)))
                    mask_img = mask_img[:,:,0]
                    
                    # Save tiles
                    #np.save(f'{tile_npy_save_dir}/{i}_{j}.npy',tile_img.astype('float32'))
                    plt.imsave(f'{tile_save_dir}/{i}_{j}.png',tile_img)
                    
                    # Save masks
                    np.save(f'{mask_save_dir}/{i}_{j}.npy',mask_img.astype('float32'))

#%% Prepare testset in NPY format

import os
import numpy as np
import staintools
from tqdm import tqdm

datatype = 'float16'

# Define stain normalization operation
def stain_norm(stain_ref,image):
    target = staintools.LuminosityStandardizer.standardize(stain_ref)
    to_transform = staintools.LuminosityStandardizer.standardize(image)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    transformed = normalizer.transform(to_transform)
    return transformed

save_dir = 'Camelyon16/Testing/Testing Set' # Directory to save the extracted tiles and their corresponding binary masks

test_case_list = os.listdir(save_dir)

stain_ref = staintools.read_image('Camelyon16/Reference Image.jpg')

for caseID in range(0,len(test_case_list)):
    cur_case = f'{save_dir}/{test_case_list[caseID]}'

    tissue_dir = f'{cur_case}/Tissue'
    mask_dir = f'{cur_case}/Mask'
    
    tissue_list = os.listdir(tissue_dir)
    
    x_test = np.zeros((len(tissue_list),512,512,3),dtype=datatype)
    y_test = np.zeros((len(tissue_list),512,512,1),dtype=datatype)
    
    for i in tqdm(range(0,len(tissue_list))):
        # Load tissue tiles and stain normalization 
        tissue_img = staintools.read_image(f'{tissue_dir}/{tissue_list[i]}')
        tissue_img_sn = stain_norm(stain_ref,tissue_img)
        
        # Load mask tiles 
        mask_img = np.load(f'{mask_dir}/{tissue_list[i].replace("png","npy")}')
        
        x_test[i,:,:,:] = tissue_img_sn
        y_test[i,:,:,:] = mask_img
    
    # Save training set and testing set
    np.save(f'{cur_case}/x_test.npy',x_test)
    np.save(f'{cur_case}/y_test.npy',y_test)