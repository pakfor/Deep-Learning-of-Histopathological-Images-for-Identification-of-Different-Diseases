# Import modules for data manipulation
import numpy as np
import os
import time 

# Import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# Enable FP16 mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

#%% Data Import

base_dir = 'E:/Camelyon16 Segmentation/Training Data'
x_train = np.load(f'{base_dir}/x_train_FP16.npy')
y_train = np.load(f'{base_dir}/y_train_FP16.npy')
x_test = np.load(f'{base_dir}/x_test_FP16.npy')
y_test = np.load(f'{base_dir}/y_test_FP16.npy')

# Make Sure All Data are Float16
x_train = x_train.astype('float16')
y_train = y_train.astype('float16')
x_test = x_test.astype('float16')
y_test = y_test.astype('float16')

# Shuffle the data
shuffler = np.random.permutation(len(x_train))
x_train = x_train[shuffler]
y_train = y_train[shuffler]
shuffler = np.random.permutation(len(x_test))
x_test = x_test[shuffler]
y_test = y_test[shuffler]

#%% Data Generator

batch_size = 4

# Define generator for training data according to batch size as defined by user
def train_gen(x_train,y_train):
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    
    while True:
        for i in range(0,len(x_train),batch_size):
            yield (x_train[i:i+batch_size],y_train[i:i+batch_size])

# Define generator for testing data, batch size is set to 1
def validate_gen(x_test,y_test):
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    y_test = y_test[shuffler]
    
    while True:
        for i in range(0,len(x_test),1):
            yield (x_test[i:i+1],y_test[i:i+1])
    
#%% Neural Network

# Define the architecture U-Net of the neural network to be trained for segmentation
def UNet():
    input = layers.Input(shape=(512,512,3))
    x = layers.Conv2D(64,3,1,padding='same')(input)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a1 = x
    
    x = layers.MaxPooling2D(2,2)(a1)
    
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a2 = x
    
    x = layers.MaxPooling2D(2,2)(a2)
    
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a3 = x
    
    x = layers.MaxPooling2D(2,2)(a3)
    
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    a4 = x
    
    x = layers.MaxPooling2D(2,2)(a4)
    
    x = layers.Conv2D(1024,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1024,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(128,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a4],axis=-1)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(64,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a3],axis=-1)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a2],axis=-1)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(16,1,2,padding='same')(x)
    
    x = layers.concatenate([x,a1],axis=-1)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,3,1,padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(1,1,1,padding='same',activation='sigmoid')(x)
    
    model = tf.keras.Model(input,x)
    model.summary()
    
    return model

#%% Model Compile & Training

# Start the timer for recording the time taken for training the model
train_start = time.time()

# Create saving paths
save_dir = 'E:/Camelyon16 Segmentation/Trained Models/20220409_ConvAE64_SN_Tr5000_Te1000_EP200_FP16_Adam(CV1)_Trial2_Tanh'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


model_save = f'{save_dir}/model' # For saving the result at the end of the training
log_save = f'{save_dir}/log' # For storing training activities
ckpt_save = f'{save_dir}/model_ckpt' # For saving the best weightings recorded during the training (continuously updated during training)

# Create the directories as mentioned above
if not os.path.exists(model_save):
    os.mkdir(model_save)
if not os.path.exists(log_save):
    os.mkdir(log_save)
if not os.path.exists(ckpt_save):
    os.mkdir(ckpt_save)

# Define optimizer: Adam
opt_adam = tf.keras.optimizers.Adam(clipvalue=1.0)

# Define callbacks: early stopping and model checkpoint
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 20, verbose = 1, mode = 'min')
model_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')

# Make model
model = UNet()

# Compile the model by defining loss function (MSE) and optimizer (Adam)
model.compile(opt_adam,loss='mean_squared_error',metrics=['accuracy'])

# Fit the model with training data and validate it with testing data every epoch
# Define maximum number of epochs
# Apply callbacks
model.fit(train_gen(x_train,y_train), steps_per_epoch = len(x_train)/batch_size, epochs = 200, validation_data = validate_gen(x_test,y_test), validation_steps = len(x_test), callbacks = [model_ckpt,earlystop])

# Stop the timer for recording the time taken for training the model
train_end = time.time()
# Calculate time taken and print it
elapse = train_end - train_start
print(elapse)

# Save the trained model to the designated directory
model.save(model_save)