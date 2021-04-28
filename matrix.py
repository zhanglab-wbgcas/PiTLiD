#%matplotlib inline
import itertools

import numpy as np
from sklearn.metrics import confusion_matrix
import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from keras import backend as K
#K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
"""
dst_path ="/home/liuk/dl/pytorch/data/corn"
val_dir= os.path.join(dst_path, 'val')
val = os.listdir(val_dir)
#model_file="/home/liuk/dl/pytorch/transfer/keras/output/weights_inceptionv3_tomato_tanh.h5"
model_file="/home/liuk/dl/pytorch/transfer/keras/output/weights_inceptionv3_potato_tanh.h5"
batch_size =15
model = load_model(model_file)

val_datagen=ImageDataGenerator(rescale=1. / 255)
val_generator =val_datagen.flow_from_directory(
    val_dir,
    target_size=(256,256),
    batch_size=batch_size,
    class_mode='categorical')
val_loss, val_acc = model.evaluate_generator(val_generator, steps=val_generator.samples / batch_size)
print('val acc: %.3f%%' % val_acc)

"""

def get_input_xy(src=[]):
    pre_x = []
    true_y = []

    #class_indices = {'gray_leaf_spot':0,'common_rust':1,'northern_Leaf_Blight':2,'healthy':3}
    #class_indices = {'black_rot':0,'cedar_rust':1,'scab':3,'healthy':2}
    class_indices = {'Early_blight':0,'healthy':1,'Late_blight':2}
    #class_indices={'bacterial':0,'healthy':1}
    #class_indices={'black_rot':0,'esca_':1,'healthy':2,'leaf_blight':3}
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



"""
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
"""
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #plt.figure(figsize=(15, 12))
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=18, pad=30,fontweight='black') #pad 参数：调节标注和图框的距离
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes) #rotation=30)
    plt.yticks(tick_marks, classes)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()#图像外部边缘的调整可以使用plt.tight_layout()进行自动控制，此方法不能够很好的控制图像间的间隔
    plt.subplots_adjust(bottom=0.2)#调整图像的整体位置

    plt.ylabel('True label',labelpad=10,fontsize=12,fontstyle='oblique')#fontproperties='simhei')
    plt.xlabel('Predicted label',labelpad=15,fontsize=12,fontstyle='oblique')
    plt.show()

#dst_path ="/home/liuk/dl/pytorch/data/newdata"
#dst_path ="/home/liuk/dl/pytorch/data/potato"
dst_path ="/home/liuk/dl/pytorch/data/potato"
test_dir=os.path.join(dst_path, 'test')
test = os.listdir(test_dir)

images = []

# 获取每张图片的地址，并保存在列表images中
for testpath in test:
    for fn in os.listdir(os.path.join(test_dir, testpath)):
        if fn.endswith('jpg'):
            fd = os.path.join(test_dir, testpath, fn)
            images.append(fd)

# 得到规范化图片及true label
pre_x, true_y = get_input_xy(images) #获得的pre_x是numpt, true_y是list
 
# 预测
#pred_y = model.predict_classes(pre_x)

#PATH = os.getcwd("/home/liuk/dl/pytorch/data/newdata")
#data_path = '/home/liuk/dl/pytorch/data/newdata/test'
#data_path = '/home/liuk/dl/pytorch/data/potato/test'
data_path = '/home/liuk/dl/pytorch/data/potato/test'
data_dir_list = os.listdir(data_path)

img_rows=256
img_cols=256
num_channel=3
#num_epoch
# Define the number of classes
num_classes = 3
#labels_name={'black_rot':0,'cedar_rust':1,'scab':3,'healthy':2}
#labels_name={'gray_leaf_spot':0,'common_rust':1,'northern_Leaf_Blight':2,'healthy':3}
#labels_name={'bacterial':0,'healthy':1}
labels_name={'Early_blight':0,'healthy':1,'Late_blight':2}
#labels_name={'black_rot':0,'esca_':1,'healthy':2,'leaf_blight':3}
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
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#model_file="/home/liuk/dl/pytorch/transfer/keras/output/weights_inceptionv3_potatoclr1e-5_tanh.h5"
model_file="/home/liuk/dl/pytorch/transfer/keras/output/weights_inceptionv3_potato_tanh.h5"

#batch_size =15
model = load_model(model_file)
pred_y = model.predict(pre_x)
pred_y=np.argmax(pred_y,axis=1) # 将one-hot转化为label
Y = np.argmax(Y ,axis=1)
# 画混淆矩阵
confusion_mat = confusion_matrix(Y, pred_y)
#attack_types = ['black_rot','cedar_rust','healthy','scab']
#attack_types = ['bacterial','healthy']
attack_types = ['Early_blight','healthy','Late_blight']
#attack_types =['black_rot','esca_','healthy','leaf_blight']
#plot_onfusion_matrix(confusion_mat, classes=range(4))
#plot_confusion_matrix(confusion_mat,range(np.max(Y)+1))
plot_confusion_matrix(confusion_mat,classes=attack_types)
plt.savefig("confmatrix/potato.jpg",format='jpg')
plt.show()

