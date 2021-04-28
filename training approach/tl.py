from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad,RMSprop,Adam,SGD
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
        '/home/liuk/dl/pytorch/data/newdata/train',
        target_size=(256,256),
        #target_size=(299,299),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
       )
val_generator = val_datagen.flow_from_directory(
        '/home/liuk/dl/pytorch/data/newdata/val',
        target_size=(256,256),
        #target_size=(299,299),
        batch_size=16,
        class_mode='categorical',
        shuffle=True)

# 构建基础模型
#base_model = InceptionV3(weights='imagenet',include_top=True)
base_model = InceptionV3(weights='imagenet',include_top=False) #include_top，如果是True，输出是1000个节点的全连接层。如果是False，会去掉顶层，输出一个8 * 8 * 2048的张量

# 增加新的输出层
x = base_model.output
x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成1xC 张量，C是通道数
x = Dense(1024,activation='relu')(x)
predictions = Dense(4,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
# plot_model(model,'tlmodel.png')

plot_model(model,'output/tlmodel.png')

'''
这里的base_model和model里面的iv3都指向同一个地址
'''

def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
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


setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=200,#800
                    epochs=50,#2
                    validation_data=val_generator,
                    validation_steps=100,#12
                    class_weight='auto'
                    )
model.save('output/tl.h5')
"""
setup_to_fine_tune(model,base_model)
history_ft = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=800,
                                 epochs=50,
                                 validation_data=val_generator,
                                 validation_steps=1,
                                 initial_epoch=10,
                                 class_weight='auto')
model.save('output/tlft10-40.h5')
"""

