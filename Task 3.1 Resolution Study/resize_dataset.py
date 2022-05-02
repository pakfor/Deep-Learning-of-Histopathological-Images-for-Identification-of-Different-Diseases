import numpy as np
from skimage.transform import resize 
from tqdm import tqdm

# Define the directory for importing original dataset (512 pix.)
import_dir = 'Resolution/Training Data/512 PIX'

# Define the directory for saving the newly processed (downsampled) dataset
save_dir = 'Resolution/Training Data/256 PIX'

original_dim = 512

# Import original x_train and x_test 
x_train_original = np.load(f'{import_dir}/x_train.npy') 
x_test_original = np.load(f'{import_dir}/x_test.npy')

# Define new dimension
new_dim = 256

# Create an array for storing the processed data
x_train_new = np.zeros((x_train_original.shape[0],new_dim,new_dim,3),dtype='float16')

for i in tqdm(range(0,x_train_original.shape[0])):
    # Read original datum
    img = x_train_original[i,:,:,:]
    img = np.resize(img,(original_dim,original_dim,3))
    img = img.astype('float32')
    
    # Resize the datum, change to appropriate datatype and save to the newly created array
    img_resized = resize(img,(new_dim,new_dim,3))
    img_resized = img_resized.astype('float16')
    
    x_train_new[i,:,:,:] = img_resized

# Create an array for storing the processed data
x_test_new = np.zeros((x_test_original.shape[0],new_dim,new_dim,3),dtype='float16')

for j in tqdm(range(0,x_test_original.shape[0])):
    # Read original datum
    img = x_test_original[j,:,:,:]
    img = np.resize(img,(original_dim,original_dim,3))
    img = img.astype('float32')
    
    # Resize the datum, change to appropriate datatype and save to the newly created array
    img_resized = resize(img,(new_dim,new_dim,3))
    img_resized = img_resized.astype('float16')
    
    x_test_new[j,:,:,:] = img_resized
    
# Save the resized, downsampled dataset to target directory
np.save(f'{save_dir}/x_train.npy',x_train_new)
np.save(f'{save_dir}/x_test.npy',x_test_new)

# y_train.npy and y_test.npy shall be copied to new directory manually