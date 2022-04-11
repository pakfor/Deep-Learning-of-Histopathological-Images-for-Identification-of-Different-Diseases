import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from skimage.transform import resize
import time 
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import TensorBoard

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

#%% Data Import

base_dir = 'E:/Camelyon16 Classification/Training Data/Original Size (512 pix)/Stain Normalized Array'
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
    
#%% Model Architecture

def resnet():
    input = layers.Input(shape=(512,512,3))
    
    x = layers.Conv2D(64,(7,7),strides=2,padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)
  
    # Conv2.x
    for i in range(0,3):
        a01 = x
        x = layers.Conv2D(64,(1,1),strides=1,padding='same')(a01)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64,(3,3),strides=1,padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256,(1,1),strides=1,padding='same')(x)
        a02 = x
        if i==0:
            a03 = layers.Conv2D(256,(1,1),strides=1,padding='same')(a01)
        else:
            a03 = a01
        x = layers.add([a02,a03])
        x = layers.ReLU()(x)
   
    # Conv3.x
    for i in range(0,4):
        b01 = x
        x = layers.Conv2D(128,(1,1),strides=1,padding='same')(b01)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if i==0:
            x = layers.Conv2D(128,(3,3),strides=2,padding='same')(x)
        else:
            x = layers.Conv2D(128,(3,3),strides=1,padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(512,(1,1),strides=1,padding='same')(x)
        b02 = x
        if i==0:
            b03 = layers.Conv2D(512,(1,1),strides=2,padding='same')(b01)
        else:
            b03 = b01
        x = layers.add([b02,b03])
        x = layers.ReLU()(x)
   
    # Conv4.x
    for i in range(0,23):
        c01 = x
        x = layers.Conv2D(256,(1,1),strides=1,padding='same')(c01)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if i==0:
            x = layers.Conv2D(256,(3,3),strides=2,padding='same')(x)
        else:
            x = layers.Conv2D(256,(3,3),strides=1,padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(1024,(1,1),strides=1,padding='same')(x)
        c02 = x
        if i==0:
            c03 = layers.Conv2D(1024,(1,1),strides=2,padding='same')(c01)
        else:
            c03 = c01
        x = layers.add([c02,c03])
        x = layers.ReLU()(x)
    
    # Conv5.x
    for i in range(0,3):
        d01 = x
        x = layers.Conv2D(512,(1,1),strides=1,padding='same')(d01)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if i==0:
            x = layers.Conv2D(512,(3,3),strides=2,padding='same')(x)
        else:
            x = layers.Conv2D(512,(3,3),strides=1,padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(2048,(1,1),strides=1,padding='same')(x)
        d02 = x
        if i==0:
            d03 = layers.Conv2D(2048,(1,1),strides=2,padding='same')(d01)
        else:
            d03 = d01
        x = layers.add([d02,d03])
        x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000)(x)
    x = layers.Dense(1,activation='sigmoid')(x)
    
    model = tf.keras.Model(input,x)
    model.summary()
    
    return model

#%% ResNet50

def resnet50():
    model = tf.keras.applications.resnet50.ResNet50(include_top=True,weights=None,input_tensor=None,input_shape=(512,512,3),pooling='average',classes=2,classifier_activation='sigmoid')

    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

    
#%% ResNeXt50

def resnet152v2():
    model = tf.keras.applications.resnet_v2.ResNet152V2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(512,512,3),
    pooling='average',
    classes=3,
    classifier_activation='sigmoid')

    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% InceptionV3

def inceptionv3():
    model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True, weights=None, input_tensor=None,
        input_shape=(512,512,3), pooling='average', classes=2,
        classifier_activation='sigmoid')
    
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model


#%% InceptionResNet2

def inception_resnet_v2():
    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights=None, input_tensor=None, input_shape=(512,512,3), pooling='average', classes=3, classifier_activation='sigmoid')
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% EfficientNetB3

def enet_b3():
    model = tf.keras.applications.EfficientNetB3(input_shape=(512,512,3),include_top=True,weights=None,input_tensor=None,pooling='average',classes=3,classifier_activation='sigmoid')
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model


#%% DenseNet

def densenet():
    model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=(512,512,3), pooling='average', classes=3)
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% NasNetLarge

def nasnetlarge():
    model = tf.keras.applications.nasnet.NASNetLarge(
    input_shape=(512,512,3),
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling='average',
    classes=3)
    
    model = tf.keras.Model(model.input,model.output)
    model.summary()
    
    return model

#%% Model Compile & Training ,

train_start = time.time()

# Create saving paths
save_dir = 'D:/Camelyon16 Classification/Trained Models/20220401_InceptionV3_SN_Tr16000_Te4000_EP200_2CL_FP16_SGD'
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

# Turn on tensorboard
#tensorboard = TensorBoard(log_dir=log_save)

# Optimizer
lr_sc = tf.keras.optimizers.schedules.ExponentialDecay(0.01, 20, 0.9, staircase=True, name=None)
opt_adam = tf.keras.optimizers.Adam(learning_rate=lr_sc)
rms_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.1,30,0.16)
opt_rms = tf.keras.optimizers.RMSprop(learning_rate=rms_lr, rho=0.9, momentum=0.9, epsilon=1.0, centered=False, name='RMSprop')

# Callbacks
earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 20, verbose = 1, mode = 'max')
model_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_save, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

model = inceptionv3() # Disable this line for on-going training activity
model.compile('SGD',loss='categorical_crossentropy',metrics='accuracy')
model.fit(train_gen(x_train,y_train), steps_per_epoch = len(x_train)/batch_size, epochs = 200, validation_data = validate_gen(x_test,y_test), validation_steps = len(x_test), callbacks = [model_ckpt,earlystop])

train_end = time.time()
elapse = train_end - train_start
print(elapse)

#%% Model Save 

model.save(model_save)