import json
import numpy as np
import os 
os.add_dll_directory('C:/Users/NGPF/anaconda3/Library/openslide-win64-20171122/bin')
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import openslide

thumbnail_store = 'E:/Camelyon16 Classification/Tumor Slides Thumbnails'
slide_store_dir = 'D:/Dataset/Open-Sourced/Camelyon16/Tumor'
list_slide = os.listdir(slide_store_dir)

if not os.path.exists(thumbnail_store):
    os.mkdir(thumbnail_store)
    
for i in range(0,len(list_slide)):
    slide = openslide.OpenSlide(f'{slide_store_dir}/{list_slide[i]}')
    thumbnail_dim = slide.level_dimensions[-1]
    #print(thumbnail_dim)
    thumbnail = np.array(slide.get_thumbnail(thumbnail_dim))
    plt.imsave(f'{slide_store_dir}/{list_slide[i].replace("tif","png")}',thumbnail)