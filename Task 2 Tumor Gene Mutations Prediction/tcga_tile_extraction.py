import numpy as np
import os 
os.add_dll_directory('C:/Users/NGPF/anaconda3/Library/openslide-win64-20171122/bin')
import openslide
from openslide import deepzoom
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

#%% Define directories

all_slide_dir = 'E:/TCGA LUAD Gene Mutation Prediction/Data/TCGA LUAD' # Directory storing all the slides of TCGA-LUAD
bg_tissue_mask_dir = 'E:/TCGA LUAD Gene Mutation Prediction/Data/TCGA LUAD Tumor Slides Thumbnail (Binary)' # Directory storing the background-tissue map (binary)
slide_mutation_file = pd.read_excel('E:/TCGA LUAD Gene Mutation Prediction/Information/Diagnostic Results/Masked Somatic Mutation/Mutation Distribution.xlsx') # Mutation information

tumor_slide_list = os.listdir(bg_tissue_mask_dir)
save_dir = 'E:/TCGA LUAD Gene Mutation Prediction/Data/TCGA LUAD Extracted Tiles (3CL Genes)'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

target_gene = ['EGFR','KRAS','STK11']

#%%

tile_dim = 512
tile_overlap = 0
bg_threshold = 0.5

for i in tqdm(range(0,len(tumor_slide_list))):
    slide_name = tumor_slide_list[i]
    slide_name = slide_name.replace('.png','')
    
    if slide_mutation_file[slide_mutation_file['Slide'] == slide_name[0:14]].shape[0] != 0: # Check that the slide has our target mutations
        
        gene_of_slide = slide_mutation_file[slide_mutation_file['Slide'] == slide_name[0:14]]['Gene Mutation'].values[0] # Obtain the exact mutation
        
        slide = openslide.OpenSlide(f'{all_slide_dir}/{slide_name[0:12]}/{slide_name}.svs') # Load tissue slide
        bg_tissue_map = plt.imread(f'{bg_tissue_mask_dir}/{slide_name}.png') # Load background-tissue map
        
        tile_gen = deepzoom.DeepZoomGenerator(slide,tile_size=tile_dim,overlap=tile_overlap,limit_bounds=False) # Set tile generator for tissues
        tile_num = tile_gen.level_tiles[-1] # at highest magnification
        tile_hor_num = tile_num[0]-1 # Number of tiles along horizontal direction
        tile_ver_num = tile_num[1]-1 # Number of tiles along vertical direction
        
        slide_dim = slide.dimensions # (horizontal pixel count, vertical pixel count)
        bg_tissue_map_dim = bg_tissue_map.shape # (vertical pixel count, horizontal pixel count)
        ratio = int(tile_dim / (int(slide_dim[0] / bg_tissue_map_dim[1]))) # Ratio for downscaling the tissue-bg map to become tile-level labeling for the tiles
        new_bg_tissue_map = np.zeros((int(bg_tissue_map_dim[0] / ratio),int(bg_tissue_map_dim[1] / ratio))) # Create new background-tissue map
        
        # Downsample the original map to new map
        for y in range(0,new_bg_tissue_map.shape[0]):
            for x in range(0,new_bg_tissue_map.shape[1]):
                sub_region = bg_tissue_map[y*ratio:(y+1)*ratio,x*ratio:(x+1)*ratio]
                new_bg_tissue_map[y,x] = np.sum(sub_region) / (ratio * ratio)
        
        for j in tqdm(range(0,tile_ver_num)):
            for i in range(0,tile_hor_num):
                if new_bg_tissue_map[j,i] >= bg_threshold: # Tiles with less than 50% of background are accepted
                    tile_img = np.array(tile_gen.get_tile(tile_gen.level_count-1,(i,j)))
                    
                    # If the tiles look pale, then it may be not tissue-containing tiles, tile score is calculated according to overall brightness
                    tile_score = np.sum(tile_img,axis=2)
                    tile_score[tile_score<220*3] = 1
                    tile_score[tile_score>=220*3] = 0
                    tile_score = np.sum(tile_score)
                    
                    if tile_score >= tile_dim * tile_dim / 2:
                        plt.imsave(f'{save_dir}/{gene_of_slide}/{slide_name}_{i}_{j}.png',tile_img)