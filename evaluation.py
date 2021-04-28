import keras
import numpy as np
import pandas as pd
import os
import random
import shutil
import cv2 # 
#import tensorflow as tf # 开源的，目前比较流行的机器学习框架
import tensorflow.compat.v1 as tf
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

#定义计算指标的metric
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1
)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro'
)
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, 
_val_precision, _val_recall))
        return
#将数据转为x_val  y_val(其中只用得到x_val数据，得到的y_val数据格式不适用于上面定义的metrics
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
        input = cv2.resize(input, (299,299))
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
                input_img_resize=cv2.resize(input_img,(299,299))
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
        #target_size=(299,299),
        target_size=(299,299),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
       )
val_generator = val_datagen.flow_from_directory(
        '/home/liuk/dl/pytorch/data/apple/val',
        #target_size=(299,299),
        target_size=(299,299),
        batch_size=16,
        class_mode='categorical',
        shuffle=True)

# 构建基础模型
#base_model = InceptionV3(weights='imagenet',include_top=True)
base_model = InceptionV3(weights=None,include_top=True) #include_top，如果是True，输出是1000个节点的全连接层。如果是False，会去掉顶层，输出一个8 * 8 * 2048的张量
"""
# 增加新的输出层
x = base_model.output
x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 
1xC 张量，C是通道数
x = Dense(1024,activation='relu')(x)
"""
x = base_model.output
x = Dense(1024,activation='relu')(x)
predictions = Dense(4,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
# plot_model(model,'tlmodel.png')

plot_model(model,'output/inceptionmodel.png')

'''
这里的base_model和model里面的iv3都指向同一个地址
'''
def setup_to_training(model,base_model):
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=RMSprop(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])


"""
def setup_to_fine_tune(model,base_model):
    
    GAP_LAYER = 17 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=RMSprop(lr=0.0001),loss='categorical_crossentropy',
metrics=['accuracy'])
"""


setup_to_training(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=200,#800
                    epochs=50,#2
                    validation_data=val_generator,
                    callbacks=[Metrics(valid_data=(pre_x,labels))],
                    validation_steps=100,#12
                    #class_weight='auto'
                    )
model.save('output/commom.h5')

