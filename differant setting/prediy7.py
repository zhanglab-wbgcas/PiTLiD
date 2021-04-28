import math

import keras
import numpy as np
import pandas as pd
import os
import random
import shutil
import cv2 # Opencv，一个强大的图像处理和计算机视觉库
import random # 用于生成各种随机数据函数
import imutils # 轻量级简化版图像处理函数
import imgaug as ia # 数据增强库，涵盖了大量对图像增强处理方法
import tensorflow as tf # 开源的，目前比较流行的机器学习框架
import imgaug.augmenters as iaa # imgaug的augmenters处理类
from keras.applications import Xception,resnet50,inception_v3,resnet
# 基于python开发的数字图片处理包
from skimage import transform
# 第三方图像处理库：图片核心处理、图片增强处理
from PIL import Image, ImageEnhance
# keras.utils中将类向量（整数）转换为one-hot编码
from keras.utils import to_categorical
# keras中内置的ImageDataGenerator图片生成器，实现图像批量增强
from keras.preprocessing.image import ImageDataGenerator
#import inception_V4
import inception_v4
from keras.callbacks import  LearningRateScheduler,TensorBoard
from keras.callbacks.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Dropout,Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.regularizers import l2,l1
from keras.applications.resnet50 import ResNet50
#from datetime import datetime
import matplotlib.pyplot as plt
import time
import datetime
from keras.optimizers import SGD
from keras_layer_normalization import LayerNormalization
import numpy as np

from clr_callback import *
from keras.optimizers import *
from keras import utils as np_utils

CATEGORIES = ['black_rot','cedar_rust','scab','healthy']
#CATEGORIES = ['Early_blight','healthy','Late_blight']
#CATEGORIES = ['gray_leaf_spot','common_rust','northern_Leaf_Blight','healthy']
#CATEGORIES = ['leaf_blight','black_rot','esca_','healthy']
#CATEGORIES = ['mosaic_virus','yellow_Leaf_Curl_Virus','target_Spot','early_blight','healthy','late_blight','leaf_Mold','Two-spotted_spider_mite','septoria_leaf_spot','bacterial_spot']
#CATEGORIES=['Early_blight','healthy','Late_blight']
#CATEGORIES=['bacterial','healthy']
print("Keras Version: ",keras.__version__)

# 时间差计算函数
def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1


def lr_schedule(epoch):
    lr = 1e-3
    #lr=10 ** uniform(-6, 1)
    if epoch > 20:
        lr *= 1e-1
    elif epoch > 40:
        lr *= 1e-2
    print('Learning rate: ', lr)
    return lr

#from imblearn.combine import SMOTEENN

print(__doc__)

def train():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        vertical_flip=True,
        horizontal_flip=True,
        #featurewise_center=True, 
        #featurewise_std_normalization=True,
        #zca_whitening=True,
        
        
        fill_mode='nearest')
    """ 
    img = load_img('/home/liuk/dl/pytorch/data/peach/train/leaf_blight/1.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory

    i = 0
    for batch in train_datagen.flow(x, batch_size=1,save_to_dir='/home/liuk/dl/pytorch/data/augpreview/peach', save_prefix='lb', save_format='jpg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely
    """
    train_generator = train_datagen.flow_from_directory(
        '/home/liuk/dl/pytorch/data/apple/train',
        #target_size=(224,224),
        target_size=(256,256),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
       )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    val_generator = val_datagen.flow_from_directory(
        '/home/liuk/dl/pytorch/data/apple/val',
        #target_size=(224,224),
        target_size=(256,256),
        batch_size=16,
        class_mode='categorical',
        shuffle=True)

    pre=pretrain_dense_layer(train_generator,val_generator)
    train_whole_model(*pre)

def pretrain_dense_layer(train_generator,val_generator):
    #since = time.time()
    model_name = "keras_inceptionv3-applevarlr_{}".format(datetime.datetime.now())
    tensorboard = TensorBoard(log_dir='output/logs/{}'.format(model_name))
    #tensorboard = TensorBoard('output/logs')
    #basic_model = inception_v3.InceptionV3(include_top=False, weights='imagenet',pooling='avg')
    basic_model = inception_v3.InceptionV3(include_top=False, weights='imagenet')
    #basic_model = resnet50.ResNet50(include_top=False, weights='imagenet',pooling='avg')
    for layer in basic_model.layers:
        layer.trainable = False

    input_tensor = basic_model.input
    x = basic_model.output
    x=GlobalAveragePooling2D()(x)
    #x = Activation('relu')(x)
    #x = Dense(1024)(x)
    #x = Dropout(.5)(x)
    #x = Activation('relu')(x)
    #x = Dense(512)(x)
    #x = Dropout(.5)(x)
    #x = Activation('relu')(x)
    # build top
    x = Dropout(.5)(x)
    x = Activation('relu')(x)
    x = Dense(len(CATEGORIES), activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(optimizer=RMSprop(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    #model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),loss='categorical_crossentropy', metrics=['accuracy'])
    from keras.callbacks import ReduceLROnPlateau,EarlyStopping
    early_stop = EarlyStopping( monitor='val_loss', patience=13, verbose=1)
    callbacks = [early_stop]
    startdate1=datetime.datetime.now() # 获取当前时间
    startdate1=startdate1.strftime("%Y-%m-%d %H:%M:%S") # 当前时间转换为指定字符串格式
    lr = LearningRateScheduler(lr_schedule)
     
    model.fit_generator(train_generator,steps_per_epoch=100, epochs=10,
                        validation_data=val_generator,
                        callbacks=[tensorboard,lr],
                        #validation_steps=284,
                        workers=1,
                        verbose=1)
    #print(time.clock() - start)
    enddate1 = datetime.datetime.now() # 获取当前时间
    enddate1 = enddate1.strftime("%Y-%m-%d %H:%M:%S") # 当前时间转换为指定字符串格式
    # 计算训练时长
    print('start date1 ',startdate1)
    print('end date1 ',enddate1)
    print('Time1 ',subtime(startdate1,enddate1)) # enddate > startdate
    return model,train_generator,val_generator,tensorboard
def train_whole_model(model,train_generator,val_generator,tensorboard):
    
    for layer in model.layers:
        #layer.W_regularizer = l2(1e-1)
        layer.W_regularizer = l2(1e-3)
        layer.trainable = True
    """
    GAP_LAYER = -34 # max_pooling_2d_2
    for layer in model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in model.layers[GAP_LAYER+1:]:
        layer.W_regularizer = l2(1e-3)
        layer.trainable = True
    """
    #clr_triangular = CyclicLR(mode='triangular')

    model.compile(optimizer=RMSprop(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=RMSprop(lr=1e-3),loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=SGD(lr_schedule(0), momentum=0.9),loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    #plot_model(model, to_file='NetStruct.png', show_shapes=True)
    from keras.callbacks import ReduceLROnPlateau,EarlyStopping
    print(os.getcwd())
    #early_stop = EarlyStopping( monitor='val_loss', patience=14, verbose=1)
    #callbacks = [early_stop]
    # call backs
    checkpointer = ModelCheckpoint(filepath="/home/liuk/dl/pytorch/transfer/keras/output/weights_inceptionv3_applevarlr_tanh.h5", verbose=1,monitor='val_accuracy', save_best_only=True,mode='max',period=1)
    #checkpointer = ModelCheckpoint(filepath='output1/weights_inceptionv3_peachclr_tanh.h5', verbose=1, save_best_only=True,period=1)
    lr = LearningRateScheduler(lr_schedule)
    #lr=1e-3
    startdate2 = datetime.datetime.now() # 获取当前时间
    startdate2 = startdate2.strftime("%Y-%m-%d %H:%M:%S") # 当前时间转换为指定字符串格式
    """
    filepath="output/weights_inceptionv3_peach_apple512_tanh.h5"
    #load params 
    if os.path.exists(filepath):
        model.load_weights(filepath)
        # 若成功加载前面保存的参数，输出下列信息
        print("checkpoint_loaded")
    """
    # train dense layer
    history=model.fit_generator(train_generator,
                        steps_per_epoch=200,
                        epochs=50,
                        validation_data=val_generator,
                        #callbacks=[checkpointer, tensorboard],
                        callbacks=[checkpointer, tensorboard, lr],
                        #callbacks=[checkpointer, tensorboard,clr_triangular],
                        initial_epoch=10,
                        #validation_steps=284,
                        workers=1,
                        verbose=1)

    enddate2 = datetime.datetime.now() # 获取当前时间
    enddate2 = enddate2.strftime("%Y-%m-%d %H:%M:%S") # 当前时间转换为指定字符串格式
    # 计算训练时长
    print('start date2 ',startdate2)
    print('end date2 ',enddate2)
    print('Time2 ',subtime(startdate2,enddate2)) # enddate > startdate

def predict():
    model=keras.models.load_model("output/weights_inceptionv3_applevarlr_tanh.h5")
    class_indices = {'black_rot':0,'scab':3,'cedar_rust':1,'healthy':2}
    #class_indices = {'black_rot':0,'esca_':1,'healthy':2,'leaf_blight':3}
    #class_indices = {'gray_leaf_spot':0,'common_rust':1,'northern_Leaf_Blight':2,'healthy':3}
    #class_indices={'Early_blight':0,'healthy':1,'Late_blight':2}
    #class_indices={'bacterial':0,'healthy':1}
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        '/home/liuk/dl/pytorch/data/apple/test',
        #target_size=(224,224),
        target_size=(256,256),
        batch_size=127,
        class_mode=None,
        shuffle=False)

    test=model.predict_generator(test_generator, steps=math.ceil(test_generator.samples*1./test_generator.batch_size),verbose=1)
    res=[list(class_indices.keys())[i] for i in np.argmax(test,axis=1)]
    imgs=[os.path.split(i)[1] for i in test_generator.filenames]
    res=pd.DataFrame(data={"file":imgs,"species":res})
    print(res.head())
    res.to_csv("output/submit_applelr.csv",index=None,sep=',')


if __name__ == '__main__':

    # split_validation_set()

    train()

    predict()

    pass

