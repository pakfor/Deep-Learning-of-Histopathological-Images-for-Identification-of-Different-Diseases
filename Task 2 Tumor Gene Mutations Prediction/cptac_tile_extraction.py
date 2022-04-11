import numpy as np
import os 
os.add_dll_directory('C:/Users/NGPF/anaconda3/Library/openslide-win64-20171122/bin')
import openslide
from openslide import deepzoom
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

#%% Dir

slide_path = 'D:/Dataset/Open-Sourced/CPTAC-LUAD/PKG - CPTAC-LUAD/LUAD'
metadata_path = 'E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Metadata'
tile_save_path = 'E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Tiles/Tumor'
thumbnail_path = 'E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Thumbnail/Binary_V2'

#%% Load Metadata & Analysis

gene_list = ['EGFR','KRAS','STK11']
gene_label = pd.read_csv(f'{metadata_path}/Case Gene Mutations.csv')
case_info = pd.read_csv(f'{metadata_path}/Case Metadata.csv')

gene_slide = {'gene':[],'slide':[]}

# Consider gene one-by-one
for gene_i in tqdm(range(0,len(gene_list))):
    gene_to_consider = gene_list[gene_i]
    list_case_valid = gene_label[gene_to_consider]
    for j in range(0,len(list_case_valid)):
        if list_case_valid[j] == 1: # Exist gene mutation
            case_name = gene_label['Case_ID'][j].replace('.','-')
            for z in range(0,len(case_info)):
                if case_info['Case_ID'][z] == case_name and case_info['Specimen_Type'][z] == 'tumor_tissue': # Only extract tiles from tumor slide
                    print(gene_to_consider, end=' ')
                    print(case_info['Slide_ID'][z])
                    gene_slide['gene'].append(gene_to_consider)
                    gene_slide['slide'].append(case_info['Slide_ID'][z])
    
gene_slide_df = pd.DataFrame(data = gene_slide)

#%% Load Relevant Slides

num_rev_slide = len(gene_slide_df)
tile_dim = 512

for i in tqdm(range(0,num_rev_slide)):
    slide = openslide.OpenSlide(f'{slide_path}/{gene_slide_df["slide"][i]}.svs')
    gene = f'{tile_save_path}/{gene_slide_df["gene"][i]}'
    tile_gen = deepzoom.DeepZoomGenerator(slide,tile_size=tile_dim,overlap=0,limit_bounds=False)
    tile_num = tile_gen.level_tiles[-1]
    tile_hor_num = tile_num[0]-1 # no boundary case
    tile_ver_num = tile_num[1]-1 # no boundary case
    
    if os.path.exists(f'{thumbnail_path}/{gene_slide_df["slide"][i]}.png'):
        # load bg-tissue mask
        mask = plt.imread(f'{thumbnail_path}/{gene_slide_df["slide"][i]}.png')
        
        # create directory for storing the tiles generated
        if not os.path.exists(f'{tile_save_path}/{gene_slide_df["slide"][i]}'): 
            os.mkdir(f'{tile_save_path}/{gene_slide_df["slide"][i]}')
        
        slide_mask_ratio = slide.dimensions[0] // mask.shape[1]
        group = int(tile_dim/slide_mask_ratio)
        bg_new_dim_x = mask.shape[1] // group
        bg_new_dim_y = mask.shape[0] // group

        new_bg_map = np.zeros((bg_new_dim_y,bg_new_dim_x))
        for u in range(0,bg_new_dim_x):
            for v in range(0,bg_new_dim_y):
                area = mask[group*v:group*(v+1),group*u:group*(u+1)]
                value = np.sum(area)/(group*group)
                new_bg_map[v,u] = value
                
        for x in range(0,tile_hor_num):
            for y in range(0,tile_ver_num):
                if new_bg_map[y,x] >= 0.5:
                    tile_img = np.array(tile_gen.get_tile(tile_gen.level_count-1,(x,y)))
                    tile_score = np.sum(tile_img,axis=2)
                    tile_score[tile_score<220*3] = 1
                    tile_score[tile_score>=220*3] = 0
                    tile_score = np.sum(tile_score)
                    if tile_score > tile_dim*tile_dim/2: 
                        plt.imsave(f'{tile_save_path}/{gene_slide_df["slide"][i]}/{x}_{y}.png',tile_img)
                        
                   
