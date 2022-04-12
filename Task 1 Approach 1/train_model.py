import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os 
import time 
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

#%% Data Import

base_dir = 'Camelyon16/Training/Training Set'
x_train = np.load(f'{base_dir}/x_train_FP16.npy')
y_train = np.load(f'{base_dir}/y_train_FP16.npy')
x_test = np.load(f'{base_dir}/x_test_FP16.npy')
y_test = np.load(f'{base_dir}/y_test_FP16.npy')

y_train = tf.keras.utils.to_categorical(y_train,num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=2)

# Make Sure All Data are Float32
x_train = x_train.astype('float16')
y_train = y_train.astype('float16')
x_test = x_test.astype('float16')
y_test = y_test.astype('float16')

x_train = x_train 
x_test = x_test

#random.seed(1)
shuffler = np.random.permutation(len(x_train))
x_train = x_train[shuffler]
y_train = y_train[shuffler]

#random.seed(2)
shuffler = np.random.permutation(len(x_test))
x_test = x_test[shuffler]
y_test = y_test[shuffler]

print(len(x_train))
print(len(x_test))


#%% Data Generator

batch_size = 16

def train_gen(x_train,y_train):
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    
    while True:
        for i in range(0,len(x_train),batch_size):
            yield (x_train[i:i+batch_size],y_train[i:i+batch_size])

def validate_gen(x_test,y_test):
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    y_test = y_test[shuffler]
    
    while True:
        for i in range(0,len(x_test),1):
            yield (x_test[i:i+1],y_test[i:i+1])
    

#%% ResNet50

def resnet50():
    model = tf.keras.applications.resnet50.ResNet50(include_top=True,weights=None,input_tensor=None,input_shape=(512,512,3),pooling='average',classes=2,classifier_activation='sigmoid')

    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% InceptionV3

def inceptionv3():
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=None,input_shape=(512,512,3), pooling='average', classes=2,classifier_activation='sigmoid')
    
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% Model Compile & Training ,

train_start = time.time()

# Create saving paths
save_dir = 'Camelyon16/Training/Training Results/Model_InceptionV3'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
model_save = f'{save_dir}/model'
log_save = f'{save_dir}/log'
ckpt_save = f'{save_dir}/model_ckpt'

if not os.path.exists(model_save):
    os.mkdir(model_save)
if not os.path.exists(log_save):
    os.mkdir(log_save)
if not os.path.exists(ckpt_save):
    os.mkdir(ckpt_save)

# Callbacks
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 20, verbose = 1, mode = 'max')
model_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_save, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

model = inceptionv3() # Disable this line for on-going training activity
model.compile('SGD',loss='binary_crossentropy',metrics='accuracy')
model.fit(train_gen(x_train,y_train), steps_per_epoch = len(x_train)/batch_size, epochs = 200, validation_data = validate_gen(x_test,y_test), validation_steps = len(x_test), callbacks = [model_ckpt,earlystop])

train_end = time.time()
elapse = train_end - train_start
print(elapse)

model.save(model_save)