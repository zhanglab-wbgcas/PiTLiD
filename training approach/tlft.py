import keras
import numpy as np
import pandas as pd
import os
import random
import shutil
import cv2 # 
#import tensorflow as tf # 开源的，目前比较流行的机器学习框架
import tensorflow.compat.v1 as tf

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
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad,RMSprop,Adam,SGD
from keras.optimizers import *
from keras import utils as np_utils
from keras.metrics import top_k_categorical_accuracy
from sklearn.metrics import f1_score, recall_score, precision_score

#from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
#from nets import inception_resnet_v2
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return
#将数据转为x_val  y_val
dst_path ="/home/liuk/dl/pytorch/data/apple"
val_dir=os.path.join(dst_path,'val')
val=os.listdir(val_dir)
images = []
# 获取每张图片的地址，并保存在列表images中
for valpath in val:
    for fn in os.listdir(os.path.join(val_dir, valpath)):
        if fn.endswith('jpg'):
            fd = os.path.join(val_dir, valpath, fn)
            images.append(fd)
#定义转变函数
def get_input_xy(src=[]):
    pre_x = []
    true_y = []

    class_indices = {'black_rot':0,'cedar_rust':1,'scab':3,'healthy':2}
    for s in src:
        input = cv2.imread(s)
        input = cv2.resize(input, (256,256))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)
        _, fn = os.path.split(s)
        y = class_indices.get(fn[:3])
        true_y.append(y)

    pre_x = np.array(pre_x) / 255.0
    return pre_x, true_y
# 得到规范化图片及true label
pre_x, true_y = get_input_xy(images)

#以下获取numpy类型的label
data_path = '/home/liuk/dl/pytorch/data/apple/val'
data_dir_list = os.listdir(data_path)

img_rows=256
img_cols=256
num_channel=3
#num_epoch
# Define the number of classes
num_classes = 4
labels_name={'black_rot':0,'cedar_rust':1,'scab':3,'healthy':2}
img_data_list=[]
labels_list = []
for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loading the images of dataset-'+'{}\n'.format(dataset))
        label = labels_name[dataset]
        for img in img_list:
                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize=cv2.resize(input_img,(256,256))
                img_data_list.append(input_img_resize)
                labels_list.append(label)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)
labels = np.array(labels_list)
print (labels.shape)
print(type(labels))


# 数据准备
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

val_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
        '/home/liuk/dl/pytorch/data/apple/train',
        target_size=(256,256),
        #target_size=(299,299),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
       )
val_generator = val_datagen.flow_from_directory(
        '/home/liuk/dl/pytorch/data/apple/val',
        target_size=(256,256),
        #target_size=(299,299),
        batch_size=16,
        class_mode='categorical',
        shuffle=True)

# 构建基础模型
base_model = InceptionV3(weights='imagenet',include_top=False,pooling='avg')
#base_model =inception_resnet_v2(weights='imagenet',include_top=False)
# 增加新的输出层
x = base_model.output
#x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = Dense(1024,activation='relu')(x)
predictions = Dense(4,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
#plot_model(model,'output/incepresnetv2.png')

def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=RMSprop(1e-3),loss='categorical_crossentropy',metrics=['accuracy']
)

def setup_to_fine_tune(model,base_model):
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=RMSprop(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=100,
                    epochs=10,#2
                    validation_data=val_generator,
                    callbacks=[Metrics(valid_data=(pre_x,labels))],
                    validation_steps=100#12
                    #class_weight='auto'
                    )

setup_to_fine_tune(model,base_model)
history_ft = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=200,
                                 epochs=50,
                                 validation_data=val_generator,
                                 callbacks=[Metrics(valid_data=(pre_x,labels))],
                                 validation_steps=100,
                                 initial_epoch=10,
                                 #class_weight='auto',
                                 )

model.save('output/tlft.h5')

