# Read downsampled masks (512x downsampled) and output an excel to extract every pixel value

import numpy as np
import os 
import pandas as pd
from tqdm import tqdm

mask_dir = 'Camelyon16/Testing/Tumor Masks (512x downsampled)' # All masks should be in numpy array format
output_dir = 'Camelyon16/Testing/Tumor Masks Pixel Information'
list_of_mask = os.listdir(mask_dir)

threshold = 0.5

for i in tqdm(range(0,len(list_of_mask))):
    record = {'x':[],'y':[],'raw_val':[],'corr_prediction':[]}
    mask = np.load(f'{mask_dir}/{list_of_mask[i]}')
    for y in range(0,mask.shape[0]):
        for x in range(0,mask.shape[1]):
            val = mask[y,x]
            record['x'].append(int(x * 512))
            record['y'].append(int(y * 512))
            record['raw_val'].append(val / 3) 
            if val / 3 >= threshold:
                corr_pred = 1
            else:
                corr_pred = 0
            record['corr_prediction'].append(corr_pred)
    record_df = pd.DataFrame(data = record)
    record_df.to_csv(f'{output_dir}/{list_of_mask[i].replace("_tumor_mask.npy",".csv")}',index=None)
    
#%% Tile-level Analysis

import numpy as np
from tqdm import tqdm
import os 
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def get_heatmap(img):
    cmap = plt.get_cmap('jet')
    rgb_img = cmap(img)
    return rgb_img[:,:,0:3]

import_dir = 'Camelyon16/Testing/Testing Set'
model_dir = 'Camelyon16/Training/Training Results/Model_InceptionV3/model_ckpt'
save_dir = 'Camelyon16/Testing/Testing Results/Model_InceptionV3'
gt_dir = 'Camelyon16/Testing/Tumor Masks Pixel Information'
gt_npy_dir = 'Camelyon16/Testing/Tumor Masks (512x downsampled)'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

record = {'case_name':[],'x':[],'y':[],'prediction_normal':[],'prediction_tumor':[],'ground_truth':[]}
test_case = os.listdir(import_dir)
num_test_case = len(test_case)

tile_dim = 512
threshold = 0.5

# Load model
model = tf.keras.models.load_model(model_dir)

for case_i in tqdm(range(0,num_test_case)):
    # Load ground truth information
    if os.path.exists(f'{gt_dir}/{test_case[case_i]}.csv'):
        gt = pd.read_csv(f'{gt_dir}/{test_case[case_i]}.csv')
        gt_slide = np.load(f'{gt_npy_dir}/{test_case[case_i]}_tumor_mask.npy')
        gt_slide = gt_slide / 3
        slide_shape = gt_slide.shape
        
        case_dir = f'{import_dir}/{test_case[case_i]}'
        tile_in_case_list = os.listdir(case_dir)
        num_tile = len(tile_in_case_list)
        
        pred_slide_th = np.zeros(slide_shape)
        pred_slide_heatmap = np.zeros(slide_shape)
        
        for tile_i in tqdm(range(0,num_tile)):
            tile_load = plt.imread(f'{case_dir}/{tile_in_case_list[tile_i]}')
            tile_load = tile_load / 255.0
            tile_load = np.reshape(tile_load,(1,512,512,3))
            
            # Get prediction
            prediction = model.predict(tile_load)
            pred_normal = float(prediction[:,0]) # Percentage to be normal
            pred_tumor = float(prediction[:,1]) # Percentage to be tumor-containing
            
            # Coordinates
            coordinates = tile_in_case_list[tile_i].replace('.jpg','')
            coor_x = int(coordinates.split('_')[0])
            coor_y = int(coordinates.split('_')[1])
            
            # Check prediction
            x_positive = gt[gt['x'] == coor_x]
            x_positive = x_positive.reset_index(drop = True)
            xy_positive = x_positive[x_positive['y'] == coor_y]
            xy_positive = xy_positive.reset_index(drop = True)
            correct_pred = int(xy_positive['corr_prediction'])
            
            record['case_name'].append(test_case[case_i])
            record['x'].append(coor_x)
            record['y'].append(coor_y)
            record['prediction_normal'].append(pred_normal)
            record['prediction_tumor'].append(pred_tumor)
            record['ground_truth'].append(correct_pred)
            
            # Construct heatmap
            if pred_tumor >= threshold:
                decision = 1 # Exist tumor
            else:
                decision = 0 # No tumor
                
            small_coor_x = int(coor_x / tile_dim)
            small_coor_y = int(coor_y / tile_dim)
            pred_slide_th[small_coor_y,small_coor_x] = decision
            pred_slide_heatmap[small_coor_y,small_coor_x] = pred_tumor
        
        gt_slide_rgb = get_heatmap(gt_slide)
        pred_slide_rgb = get_heatmap(pred_slide_heatmap)
        
        plt.imshow(gt_slide_rgb)
        plt.show()
        plt.imshow(pred_slide_rgb)
        plt.show()
        
        slide_save = f'{save_dir}/{test_case[case_i]}'
        if not os.path.exists(slide_save):
            os.mkdir(slide_save)
        
        plt.imsave(f'{slide_save}/gt_slide_threshold.png',gt_slide,cmap='gray')
        plt.imsave(f'{slide_save}/gt_slide_heatmap.png',gt_slide_rgb)
        plt.imsave(f'{slide_save}/pred_slide_threshold.png',pred_slide_th,cmap='gray')
        plt.imsave(f'{slide_save}/pred_slide_heatmap.png',pred_slide_rgb)
        
record_df = pd.DataFrame(data = record)
record_df.to_csv(f'{save_dir}/record.csv',index=None)
