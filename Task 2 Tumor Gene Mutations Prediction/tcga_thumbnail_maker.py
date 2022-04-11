import json
import numpy as np
import os 
os.add_dll_directory('C:/Users/NGPF/anaconda3/Library/openslide-win64-20171122/bin')
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import openslide

#%%

f = open('E:/TCGA LUAD Gene Mutation Prediction/Information/TCGA-LUAD slide information.json')
data = json.load(f)

tumor_slide_list = []

for i in range(0,len(data)):
    
    if data[i]['cases'][0]['samples'][0]['sample_type'] != 'Solid Tissue Normal':
        tumor_slide_list.append(data[i]['file_name'].split('.')[0])

#%% Thumbnail

thumbnail_store = 'E:/TCGA LUAD Gene Mutation Prediction/Data/TCGA LUAD Tumor Slides Thumbnail'
slide_store_dir = 'E:/TCGA LUAD Gene Mutation Prediction/Data/TCGA LUAD'

for i in tqdm(range(0,len(tumor_slide_list))):
    try:
        slide = openslide.OpenSlide(f'{slide_store_dir}/{tumor_slide_list[i][0:12]}/{tumor_slide_list[i]}.svs')
        thumbnail_dim = slide.level_dimensions[-1]
        print(thumbnail_dim)
        thumbnail = np.array(slide.get_thumbnail(thumbnail_dim))
        #plt.imshow(thumbnail)
        plt.imsave(f'{thumbnail_store}/{tumor_slide_list[i]}.png',thumbnail)
    
    except:
        print('No such slide')