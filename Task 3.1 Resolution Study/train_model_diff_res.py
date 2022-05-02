import numpy as np
import tensorflow as tf
import os 
import time 
from tensorflow.keras import mixed_precision

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

#%% Data Import

# Define the resolution of the training data
res = 256

# Load numpy array
base_dir = f'Resolution/Training Data/{res} PIX'
x_train = np.load(f'{base_dir}/x_train_FP16.npy')
y_train = np.load(f'{base_dir}/y_train_FP16.npy')
x_test = np.load(f'{base_dir}/x_test_FP16.npy')
y_test = np.load(f'{base_dir}/y_test_FP16.npy')

# Convert labels to categotical 
y_train = tf.keras.utils.to_categorical(y_train,num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=3)

# Make sure all data are FP16
x_train = x_train.astype('float16')
y_train = y_train.astype('float16')
x_test = x_test.astype('float16')
y_test = y_test.astype('float16')

# Shuffle training dataset
shuffler = np.random.permutation(len(x_train))
x_train = x_train[shuffler]
y_train = y_train[shuffler]

# Shuffle testing dataset
shuffler = np.random.permutation(len(x_test))
x_test = x_test[shuffler]
y_test = y_test[shuffler]

#%% Data Generator

# Define batch size
batch_size = 16

# Define data generator for training set according to batch size (user-defined)
def train_gen(x_train,y_train):
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    
    while True:
        for i in range(0,len(x_train),batch_size):
            yield (x_train[i:i+batch_size],y_train[i:i+batch_size])

# Define data generator for testing set according to batch size = 1
def validate_gen(x_test,y_test):
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    y_test = y_test[shuffler]
    
    while True:
        for i in range(0,len(x_test),1):
            yield (x_test[i:i+1],y_test[i:i+1])

#%% InceptionResNet2

# Define InceptionResNetV2 architecture
def inception_resnet_v2():
    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, input_tensor=None, input_shape=(res,res,3), pooling='average', classes=3, classifier_activation='sigmoid')
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% Model Compile & Training

model = inception_resnet_v2() # Define model architecture to be trained

# Start timer to record the start of training
train_start = time.time()

# Create saving paths, can be modified to desired locations
save_dir = f'Resolution/Trained Models/InceptionResNetV2_{res} PIX'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model_save = f'{save_dir}/model' # For saving the weightings after the last epoch
log_save = f'{save_dir}/log' # For saving training information
ckpt_save = f'{save_dir}/model_ckpt' # For saving model checkpoint, which is the best weightings recorded during training

# Create paths for saving if they do not exist
if not os.path.exists(model_save):
    os.mkdir(model_save)
if not os.path.exists(log_save):
    os.mkdir(log_save)
if not os.path.exists(ckpt_save):
    os.mkdir(ckpt_save)

# Optimizer
opt_sgd = tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.0,nesterov=False)

# Callbacks (early stopping & model checkpoint)
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.001, patience = 20, verbose = 1, mode = 'max')
model_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_save, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

model.compile(opt_sgd,loss='categorical_crossentropy',metrics='accuracy') # Compile model by giving optimizer and loss function
model.fit(train_gen(x_train,y_train), steps_per_epoch = len(x_train)/batch_size, epochs = 100, validation_data = validate_gen(x_test,y_test), validation_steps = len(x_test), callbacks = [earlystop,model_ckpt]) # Fit the model by training data and apply callbacks

# Stop timer and calculate time taken
train_end = time.time()
elapse = train_end - train_start
print(elapse)

# Save the model's weightings after the last epoch
model.save(model_save)