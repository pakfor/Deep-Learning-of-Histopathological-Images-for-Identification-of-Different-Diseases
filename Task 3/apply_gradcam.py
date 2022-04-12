import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_keras_vis.utils import num_of_gpus
from tensorflow.keras import mixed_precision
from tqdm import tqdm
import pandas as pd
import cv2
import os

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

test_set_name = 'TCGA-LUAD' # Choose from [TCGA-LUAD/CPTAC-LUAD]
method = 'GradCAM' # Choose from [GradCAM/GradCAM++]

model = tf.keras.models.load_model('GradCAM/Trained Models')
model.summary()

# CPTAC_LUAD
test_set = np.load(f'GradCAM/Testing/Testing Set/{test_set_name}/x_test.npy')

result_file = pd.read_csv(f'GradCAM/Model Performance/20220404_InceptionResNetV2_SN_Tr14400_Te3600_EP200_SGD_Binary_3CL_FP16_results_{test_set_name}.csv')

# TCGA-LUAD
#test_set = np.load('E:/Deep Learning/Final Year Project/GradCAM/Testing Sets/TCGA-LUAD/x_test.npy')
#result_file = pd.read_csv('E:/Deep Learning/Final Year Project/GradCAM/Results/20220404_InceptionResNetV2_SN_Tr14400_Te3600_EP200_SGD_Binary_3CL_FP16_results_TCGA-LUAD.csv')

save_dir = f'GradCAM/Output/20220404_InceptionResNetV2_SN_Tr14400_Te3600_EP200_SGD_Binary_3CL_FP16/{method}/{test_set_name}'

tissue_save_dir = f'GradCAM/Output/20220404_InceptionResNetV2_SN_Tr14400_Te3600_EP200_SGD_Binary_3CL_FP16/Ground Truth/{test_set_name}'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(tissue_save_dir):
    os.mkdir(tissue_save_dir)
    
if not os.path.exists(f'{save_dir}/EGFR'):
    os.mkdir(f'{save_dir}/EGFR')
if not os.path.exists(f'{save_dir}/KRAS'):
    os.mkdir(f'{save_dir}/KRAS')
if not os.path.exists(f'{save_dir}/STK11'):
    os.mkdir(f'{save_dir}/STK11')
if not os.path.exists(f'{tissue_save_dir}/EGFR'):
    os.mkdir(f'{tissue_save_dir}/EGFR')
if not os.path.exists(f'{tissue_save_dir}/KRAS'):
    os.mkdir(f'{tissue_save_dir}/KRAS')
if not os.path.exists(f'{tissue_save_dir}/STK11'):
    os.mkdir(f'{tissue_save_dir}/STK11')
    
# Define function

# Generate heatmap from grayscale image
def grayscale2rgb(img):
    cmap = plt.get_cmap('jet')
    rgb_img = cmap(img)
    return rgb_img[:,:,0:3]

# Change the activation of the last layer in the neural network to linear
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear

#%% Select tiles that were classified with high confidence: > 0.95, others < 0.05

threshold_1 = 0.95 # Higher than threshold_1 = high confidence
threshold_2 = 0.05 # Lower than threshold_2 = low_confidence

EGFRTiles = {'index':[],'EGFR':[],'KRAS':[],'STK11':[],'ground truth':[]}
KRASTiles = {'index':[],'EGFR':[],'KRAS':[],'STK11':[],'ground truth':[]}
STK11Tiles = {'index':[],'EGFR':[],'KRAS':[],'STK11':[],'ground truth':[]}

# Extract information of tiles with high confidence on correct labels while low confidence on incorrect labels
for i in range(0,len(result_file)):
    
    if result_file['Ground Truth'][i] == 0 and result_file['EGFR'][i] >= threshold_1 and result_file['KRAS'][i] < threshold_2 and result_file['STK11'][i] < threshold_2:
        EGFRTiles['index'].append(i+1)
        EGFRTiles['EGFR'].append(result_file['EGFR'][i])
        EGFRTiles['KRAS'].append(result_file['KRAS'][i])
        EGFRTiles['STK11'].append(result_file['STK11'][i])
        EGFRTiles['ground truth'].append(result_file['Ground Truth'][i])
        
    if result_file['Ground Truth'][i] == 1 and result_file['KRAS'][i] >= threshold_1 and result_file['EGFR'][i] < threshold_2 and result_file['STK11'][i] < threshold_2:
        KRASTiles['index'].append(i+1)
        KRASTiles['EGFR'].append(result_file['EGFR'][i])
        KRASTiles['KRAS'].append(result_file['KRAS'][i])
        KRASTiles['STK11'].append(result_file['STK11'][i])
        KRASTiles['ground truth'].append(result_file['Ground Truth'][i])

    if result_file['Ground Truth'][i] == 2 and result_file['STK11'][i] >= threshold_1 and result_file['EGFR'][i] < threshold_2 and result_file['KRAS'][i] < threshold_2:
        STK11Tiles['index'].append(i+1)
        STK11Tiles['EGFR'].append(result_file['EGFR'][i])
        STK11Tiles['KRAS'].append(result_file['KRAS'][i])
        STK11Tiles['STK11'].append(result_file['STK11'][i])
        STK11Tiles['ground truth'].append(result_file['Ground Truth'][i])
        
        
#%% Sort and Filter the Data

# EGFR = 0, KRAS = 1, STK11 = 2
relTilesList = []
relTilesLabelList = []

# EGFR
for i in range(0,len(EGFRTiles['index'])):
    tileLoad = test_set[int(EGFRTiles['index'][i]),:,:,:]
    relTilesList.append(np.array(tileLoad))
    relTilesLabelList.append(0)

for i in range(0,len(KRASTiles['index'])):
    tileLoad = test_set[int(KRASTiles['index'][i]),:,:,:]
    relTilesList.append(np.array(tileLoad))
    relTilesLabelList.append(1)
    
for i in range(0,len(STK11Tiles['index'])):
    tileLoad = test_set[int(STK11Tiles['index'][i]),:,:,:]
    relTilesList.append(np.array(tileLoad))
    relTilesLabelList.append(2)

for imgI in tqdm(range(0,len(relTilesList))):
    
    images = np.asarray(relTilesList[imgI],dtype='float16')
    score = CategoricalScore(relTilesLabelList[imgI])

    # Create Gradcam object
    if method == 'GradCAM':
        gradcam = Gradcam(model,model_modifier=model_modifier_function(model),clone=True)
        cam = gradcam(score,images,penultimate_layer=-1)
    else:
        gradcampp = GradcamPlusPlus(model,model_modifier=model_modifier_function(model),clone=True)
        cam = gradcampp(score,images,penultimate_layer=-1)
    
    #scorecam = Scorecam(model,model_modifier=model_modifier_function(model))
    #cam = scorecam(score,images,penultimate_layer=-1,max_N=10)
    
    # Blend the generated heatmap from GradCAM and the original tissue image
    img = cv2.addWeighted(grayscale2rgb(np.reshape(cam,(512,512))).astype('float32'),0.5,relTilesList[imgI].astype('float32'),0.5,0)
    plt.imshow(img)
    plt.show()
    
    if imgI < len(EGFRTiles['index']):
        file_name = str(len(os.listdir(f'{save_dir}/EGFR'))).zfill(3)
        plt.imsave(f'{save_dir}/EGFR/{file_name}.png',img)
        
        file_name = str(len(os.listdir(f'{tissue_save_dir}/EGFR'))).zfill(3)
        plt.imsave(f'{tissue_save_dir}/EGFR/{file_name}.png',relTilesList[imgI].astype('float32'))
        
    elif len(EGFRTiles['index']) <= imgI < len(EGFRTiles['index']) + len(KRASTiles['index']):
        file_name = str(len(os.listdir(f'{save_dir}/KRAS'))).zfill(3)
        plt.imsave(f'{save_dir}/KRAS/{file_name}.png',img)
        
        file_name = str(len(os.listdir(f'{tissue_save_dir}/KRAS'))).zfill(3)
        plt.imsave(f'{tissue_save_dir}/KRAS/{file_name}.png',relTilesList[imgI].astype('float32'))
        
    else:
        file_name = str(len(os.listdir(f'{save_dir}/STK11'))).zfill(3)
        plt.imsave(f'{save_dir}/STK11/{file_name}.png',img)
        
        file_name = str(len(os.listdir(f'{tissue_save_dir}/STK11'))).zfill(3)
        plt.imsave(f'{tissue_save_dir}/STK11/{file_name}.png',relTilesList[imgI].astype('float32'))