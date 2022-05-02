import numpy as np
import pandas as pd
import os 
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

#%% import model

trained_model_path_all = 'Resolution/Trained Models' # The saving path of trained models
model_list = os.listdir(trained_model_path_all) # List all the trained models in the saving path
for i in range(0,len(model_list)): # Print the list of models 
    print(i,model_list[i])
selection = input('Select the model to validate: ') # Ask for user input to choose a model to evaluate
selection = int(selection)

dim = input('Dimension of input: ') # Ask for user input to choose a model to evaluate
dim = int(dim)

trained_model_path = f'{trained_model_path_all}/{model_list[selection]}/model_ckpt'
trained_model = tf.keras.models.load_model(trained_model_path) # Load the selected model
print(trained_model_path,'is selected.')

dataset = input('Select dataset: [TCGA-LUAD/CPTAC-LUAD]') # Ask for user input to either use testset from TCGA-LUAD or CPTAC-LUAD

#%% import testing dataset (TCGA)

# Load TCGA testset 
if dataset == 'TCGA-LUAD':
    data_path = f'Resolution/Validation Data/{dataset}/Data ({dim} PIX)'
    x_test = np.load(f'{data_path}/x_test.npy')
    y_test = np.load(f'{data_path}/y_test.npy')

#%% import testing dataset (CPTAC)

# Load CPTAC testset 
if dataset == 'CPTAC-LUAD':
    data_path = f'Resolution/Validation Data/{dataset}/Data ({dim} PIX)'
    x_test = np.load(f'{data_path}/x_test.npy')
    y_test = np.load(f'{data_path}/y_test.npy')

#%% make prediction

# Collect predictions made by the model based on each tile, and obtain groundtruth 
results = {'EGFR':[],'KRAS':[],'STK11':[],'Ground Truth':[]}
for i in tqdm(range(0,x_test.shape[0])):    
    result = trained_model.predict(x_test[i,:,:,:].reshape(1,dim,dim,3))
    results['EGFR'].append(result[0,0])
    results['KRAS'].append(result[0,1])
    results['STK11'].append(result[0,2])
    results['Ground Truth'].append(y_test[i,0])
    
#%%

# Save the prediction results
results_df = pd.DataFrame(data = results)
print(results_df)
save_path = f'Resolution/Model Results/{model_list[selection]}_results_{dataset}.csv'

if os.path.exists(save_path):
    choice = input('This model may have been tested, continue to save the result? [Yes/No]')
    if choice == 'Yes':
        results_df.to_csv(save_path,index = False)
    else:
        print('Result is not saved, this program is terminated.')
    
else: 
    results_df.to_csv(save_path,index = False)