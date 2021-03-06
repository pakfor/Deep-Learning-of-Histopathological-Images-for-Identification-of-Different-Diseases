import numpy as np
import os 
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt

#%% Dir 

slide_path = 'CPTAC-LUAD/Data/Slides'
slide_list = os.listdir(slide_path)
print(slide_list)

#%% Make Thumbnail

thumbnail_save = 'CPTAC-LUAD/Data/Thumbnails'

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
    plt.imsave(f'{thumbnail_save}/{slide_list[i].replace("svs","")}.png',thumbnail_np)
    