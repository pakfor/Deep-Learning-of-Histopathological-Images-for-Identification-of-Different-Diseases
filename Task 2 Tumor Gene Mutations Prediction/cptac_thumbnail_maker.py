import numpy as np
import os 
os.add_dll_directory('C:/Users/NGPF/anaconda3/Library/openslide-win64-20171122/bin')
import openslide
from openslide import deepzoom
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt

#%% Dir 

slide_path = 'D:/Dataset/Open-Sourced/CPTAC-LUAD/PKG - CPTAC-LUAD/LUAD'
slide_list = os.listdir(slide_path)
print(slide_list)

#%% Make Thumbnail

num_slide = len(slide_list)
for i in tqdm(range(0,num_slide)):
    slide = openslide.OpenSlide(f'{slide_path}/{slide_list[i]}')
    try:
        scaled_dim = slide.level_dimensions[2]
    except:
        scaled_dim = slide.level_dimensions[-1]
    print(scaled_dim)
    thumbnail = slide.get_thumbnail(scaled_dim)
    thumbnail_np = np.array(thumbnail)
    plt.imsave(f'E:/TCGA LUAD Gene Mutation Prediction/Validation (CPTAC-LUAD)/Data/Thumbnail/{slide_list[i].replace("svs","")}.png',thumbnail_np)
    