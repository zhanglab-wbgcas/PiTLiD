#全部网络的迁移学习，网络定义加上自己的训练代码
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from torch.optim import lr_scheduler
from PIL import Image
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "/home/liuk/dl/pytorch/data/newdata"

# Number of classes in the dataset
num_classes = 4 #两类数据1，2

# Batch size for training (change depending on how much memory you have)
batch_size = 16 #batchsize尽量选取合适，否则训练时会内存溢出
# Number of epochs to train for
num_epochs = 50


# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

Loss_list = []
Accuracy_list = []

# 训练与验证网络（所有层都参加训练）
def train_model(model,dataloaders, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    # 保存网络训练最好的权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每训练一个epoch，测试一下网络模型的准确率
        for phase in ['train', 'val']:
            if phase == 'train':
                # 学习率更新方式
                scheduler.step()
                #  调用模型训练
                model.train(True)
            else:
                # 调用模型测试
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            # 依次获取所有图像，参与模型训练或测试
            for data in dataloaders[phase]:
                # 获取输入
                inputs, labels = data
                # 判断是否使用gpu
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # 梯度清零
                optimizer.zero_grad()

                # 网络前向运行
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                #model_name是inception时，要换成这个,经实践，还是会报错
                #_, preds = torch.max(outputs,1)
                # 计算Loss值
                #loss = criterion(outputs, labels)
                loss = criterion(outputs, labels)

                # 反传梯度，更新权重
                if phase == 'train':
                    # 反传梯度
                    loss.backward()
                    # 更新权重
                    optimizer.step()
                    # scheduler.step()
                # 计算一个epoch的loss值和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # 计算Loss和准确率的均值
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}%'.format(
                phase, epoch_loss, epoch_acc * 100))

            # 保存测试阶段，准确率最高的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                Accuracy_list.append(epoch_acc* 100)
                Loss_list.append(epoch_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}%'.format(best_acc * 100))
    # 网络导入最好的网络权重
    model.load_state_dict(best_model_wts)
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18"""
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet"""
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg16":
            """ VGG16
            """
            model_ft = models.vgg16(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "resnet50":
            """resnet50"""
            model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3 
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "inception"  # 调用模型的名字
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

"""
#是否加载之前训练过的模型
we='/home/liuk/dl/pytorch/transfer/alexnet_pytorch.pth'
model_ft.load_state_dict(torch.load(we))
"""
# Print the model we just instantiated
print(model_ft, input_size)


# 是否使用gpu运算
use_gpu = torch.cuda.is_available()
# Send the model to GPU
if use_gpu:
        model = model_ft.cuda()
"""
#自定义transform
class Cutout(object):
    def __init__(self, hole_size):
        # 正方形马赛克的边长，像素为单位
        self.hole_size = hole_size

    def __call__(self, img):
        return cutout(img, self.hole_size)

def cutout(img, hole_size):
    y = np.random.randint(32)
    x = np.random.randint(32)

    half_size = hole_size // 2

    x1 = np.clip(x - half_size, 0, 32)
    x2 = np.clip(x + half_size, 0, 32)
    y1 = np.clip(y - half_size, 0, 32)
    y2 = np.clip(y + half_size, 0, 32)

    imgnp = np.array(img)

    imgnp[y1:y2, x1:x2] = 0
    img = Image.fromarray(imgnp.astype('uint8')).convert('RGB')
    return img
"""
# 准备数据
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #Cutout(6),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
    ['train', 'val']}
# 读取数据集大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#数据类别
class_names = image_datasets['train'].classes

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.95)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,exp_lr_scheduler, num_epochs=num_epochs)
                             #,is_inception=(model_name == "inception"))

torch.save(model_ft.state_dict(), "model_inceptionfc.pkl")


x1 = range(0, 50)
x2 = range(0, 50)
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('val accuracy vs. epoches')
plt.ylabel('val accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('val loss vs. epoches')
plt.ylabel('val loss')
plt.show()
plt.savefig("inceptionresult/onlyfc_accuracy_loss.jpg")


