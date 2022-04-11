import numpy as np
import os 
os.add_dll_directory('C:/Users/NGPF/anaconda3/Library/openslide-win64-20171122/bin')
import openslide
from openslide import deepzoom
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

save_dir = 'E:/Camelyon16 Segmentation/Testing/Slide-level Segmentation Maps (512x downsampled)'
ground_truth_slide_dir = 'E:/Camelyon16 Segmentation/Testing/Slide-level Segmentation Maps'
list_slide = os.listdir(ground_truth_slide_dir)
tile_dim = 512

for i in tqdm(range(0,len(list_slide))):
    gt_slide = openslide.OpenSlide(f'{ground_truth_slide_dir}/{list_slide[i]}')
    tile_gen = deepzoom.DeepZoomGenerator(gt_slide,tile_size=tile_dim,overlap=0,limit_bounds=False)
    tile_num = tile_gen.level_tiles[-1]
    tile_hor_num = tile_num[0]-1 # no boundary case
    tile_ver_num = tile_num[1]-1 # no boundary case
    
    new_map = np.zeros((tile_ver_num,tile_hor_num))
    for y in tqdm(range(0,tile_ver_num)):
        for x in range(0,tile_hor_num):
            tile_img = np.array(tile_gen.get_tile(tile_gen.level_count-1,(x,y)))
            pixel_val = np.sum(tile_img) / (tile_dim * tile_dim)
            new_map[y,x] = pixel_val
    
    np.save(f'{save_dir}/{list_slide[i].replace(".tif",".npy")}',new_map)
    plt.imshow(new_map,cmap='gray')
    plt.show()