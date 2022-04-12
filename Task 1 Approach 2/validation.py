import numpy as np
import os 
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Define function that calculates mean squared error between 2 images (binary)
def calMSE(prediction,ground_truth):
    mse = (np.sum((prediction - ground_truth)**2)) / (512*512)
    return mse

# Define function that transforms grayscale map into heatmap
def get_heatmap(img):
    cmap = plt.get_cmap('jet') # Implement jet colorbar 
    rgb_img = cmap(img)
    return rgb_img[:,:,0:3]

# Define basic directories
test_set_dir = 'Camelyon16/Testing/Testing Set'
model_to_be_tested = tf.keras.models.load_model('Camelyon16/Training/Training Results/Model_UNet/model_ckpt')
gt_mask_all = 'Camelyon16/Testing/Tumor Masks (512x downsampled)'
save_dir = 'Camelyon16/Testing/Testing Results/Model_UNet'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# List of case to test with
list_ts_case = os.listdir(test_set_dir)

record = {'Case Number':[],'x':[],'y':[],'MSE':[],'Prediction Averaged Tile Value':[],'Ground-truth Averaged Tile Value':[]}

for i in tqdm(range(0,len(list_ts_case))):
    cur_case_dir = f'{test_set_dir}/{list_ts_case[i]}'
    tissue_dir = f'{cur_case_dir}/Tissue (Stain Norm)'
    mask_dir = f'{cur_case_dir}/Mask'
    
    list_tissue = os.listdir(tissue_dir)
    list_mask = os.listdir(mask_dir)
    
    gt_mask = np.load(f'{gt_mask_all}/{list_ts_case[i]}_tumor_mask.npy') # Import ground-truth to get the size of slide 
       
    case_save = f'{save_dir}/{list_ts_case[i]}'
    
    # Create saving directory for the case tested
    if not os.path.exists(case_save):
        os.mkdir(case_save)
    
    case_save_img = f'{case_save}/Images'
    
    # Create saving directory for image output mask for the case tested
    if not os.path.exists(case_save_img):
        os.mkdir(case_save_img)
    
    case_save_sl = f'{case_save}/Slides'
    
    # Create saving directory for slide output mask for the case tested
    if not os.path.exists(case_save_sl):
        os.mkdir(case_save_sl)
    
    heatmap_bi = np.zeros(gt_mask.shape)
    
    for j in tqdm(range(0,len(list_tissue))):
        if os.path.exists(f'{mask_dir}/{list_tissue[j].replace("png","npy")}'): # If both tissue tile and mask tile exist
            img_to_test = plt.imread(f'{tissue_dir}/{list_tissue[j]}')
            mask_as_gt = np.load(f'{mask_dir}/{list_tissue[j].replace("png","npy")}')
            
            img_to_test = img_to_test[:,:,0:3] # PNG image has 4 color channels
            img_to_test = np.reshape(img_to_test,(1,512,512,3)) # Batch size = 1
            img_to_test = img_to_test.astype('float16') # The model only accepts FP16 input
            
            pred_mask = model_to_be_tested.predict(img_to_test)
            
            pred_mask = np.reshape(pred_mask,(512,512))
            pred_mask = pred_mask.astype('float64')
            mask_as_gt = mask_as_gt.astype('float64')
            mse = calMSE(pred_mask,mask_as_gt) # Calculate MSE between the prediction as ground-truth
            
            pred_mask = pred_mask.astype('float32')
            mask_as_gt = mask_as_gt.astype('float32')
            
            # Get the coordinates of the predicted tiles
            x_coor = int(list_tissue[j].replace(".png","").split('_')[0])
            y_coor = int(list_tissue[j].replace(".png","").split('_')[1])
            
            # Save the prediction
            plt.imsave(f'{case_save_img}/{x_coor}_{y_coor}.png',pred_mask,cmap='gray',vmin=0,vmax=1)
            
            # Calculate the pixel-averaged of the prediction and append this value to construct the heat map
            heatmap_bi[y_coor,x_coor] = np.sum(pred_mask) / (512 * 512)
            
            record['Case Number'].append(list_ts_case[i])
            record['x'].append(x_coor)
            record['y'].append(y_coor)
            record['MSE'].append(mse)
            record['Prediction Averaged Tile Value'].append(np.sum(pred_mask) / (512 * 512))
            record['Ground-truth Averaged Tile Value'].append(np.sum(mask_as_gt) / (512 * 512))
    
    plt.imsave(f'{case_save}/heatmap_binary.png',heatmap_bi,cmap='gray')
    
    heatmap = get_heatmap(heatmap_bi)
    plt.imsave(f'{case_save}/heatmap.png',heatmap)
    

record_df = pd.DataFrame(data = record)
record_df.to_csv(f'{save_dir}/results.csv',index = None)