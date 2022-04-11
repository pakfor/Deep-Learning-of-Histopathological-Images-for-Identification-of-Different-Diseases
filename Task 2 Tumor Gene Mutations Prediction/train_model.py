import numpy as np
import tensorflow as tf
import os 
import time 
from tensorflow.keras import mixed_precision

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

#%% Data Import

# Load numpy array
base_dir = 'E:/Deep Learning/Final Year Project/Gene Mutation Detection/TCGA-LUAD (Training)/ArrayV1'
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
batch_size = 8

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

#%% ResNet50

# Define ResNet50 architecture
def resnet50():
    model = tf.keras.applications.resnet50.ResNet50(include_top=True,weights=None,input_tensor=None,input_shape=(512,512,3),pooling='average',classes=3,classifier_activation='sigmoid')

    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% InceptionV3

# Define InceptionV3 architecture
def inceptionv3():
    model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True, weights=None, input_tensor=None,
        input_shape=(512,512,3), pooling='average', classes=3,
        classifier_activation='sigmoid')
    
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% InceptionResNet2

# Define InceptionResNetV2 architecture
def inception_resnet_v2():
    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, input_tensor=None, input_shape=(512,512,3), pooling='average', classes=3, classifier_activation='sigmoid')
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% DenseNet

# Define DenseNet121 architecture
def densenet():
    model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=(512,512,3), pooling='average', classes=3)
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% Model Compile & Training

# Start timer to record the start of training
train_start = time.time()

# Create saving paths, can be modified to desired locations
save_dir = 'E:/Deep Learning/Final Year Project/Gene Mutation Detection/Trained Models/20220329_ResNet50_SN_Tr14400_Te3600_EP200_3CL_FP16_SGD'
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

# Callbacks (early stopping & model checkpoint)
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 10, verbose = 1, mode = 'max')
model_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_save, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

model = resnet50() # Define model architecture to be trained
model.compile('SGD',loss='categorical_crossentropy',metrics='accuracy') # Compile model by giving optimizer and loss function
model.fit(train_gen(x_train,y_train), steps_per_epoch = len(x_train)/batch_size, epochs = 100, validation_data = validate_gen(x_test,y_test), validation_steps = len(x_test), callbacks = [earlystop,model_ckpt]) # Fit the model by training data and apply callbacks

# Stop timer and calculate time taken
train_end = time.time()
elapse = train_end - train_start
print(elapse)

# Save the model's weightings after the last epoch
model.save(model_save)