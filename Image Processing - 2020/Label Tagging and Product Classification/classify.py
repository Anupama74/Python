# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:44:09 2019

@author: anupamasj
"""

#%matplotlib inline
import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn


from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,roc_curve,auc

from PIL import Image
from PIL import Image as pil_image
from PIL import ImageDraw

from time import time
from glob import glob
from tqdm import tqdm
#from skimage.io import imread
from IPython.display import SVG

from scipy import misc,ndimage
from scipy.ndimage.interpolation import zoom
#from scipy.ndimage import imread


from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler


#import images
##import labels
imagesPath = r'D:\Product Classification\Grocery-Product-Classification-master'
#r'D:\Product Classification\ClothingAttributeDataset\Apparels'
# r'D:\Product Classification\Grocery-Product-Classification-master'
#

def getAllDir (folderPath):
       getdir = []
       for root, dirs, files in os.walk(folderPath):
              for label in files:
                     with open(os.path.join(root, label)) :
                            pass
                     
              getdir.append(root)
       dirs = getdir[1:]
       return dirs

def getLabels (folderPath):
       getlabels =[]
       for root, dirs, files in os.walk(imagesPath):
              for label in dirs:
                     getlabels.append(label)
       return getlabels

       
def label_assignment(img,label):
    return label


def trainingData(label,data_dir):
    for img in tqdm(os.listdir(data_dir)):
        label = label_assignment(img,label)
        path = os.path.join(data_dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(imgsize,imgsize))
        
        X.append(np.array(img))
        Z.append(str(label))      
X = []
Z = []
imgsize = 150
##       
print(getAllDir(imagesPath))   
#print(getLabels(imagesPath)[0])

n =len(getLabels(imagesPath))
#
x=2
#for i in range(0,x):
#       trainingData(getLabels(imagesPath)[i],getAllDir(imagesPath)[i])
#trainingData(getLabels(imagesPath)[0],getAllDir(imagesPath)[0])
##############_____________________________________________
label_encoder= LabelEncoder()
Y = label_encoder.fit_transform(Z)
Y = to_categorical(Y,n)
X = np.array(X)
X=X/255
#print(len(X))
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.40,random_state=42)
                                                             ###0.25

print(len(x_train),len(x_test))
print(len(y_train), len(y_test))

augs_gen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2, 
        horizontal_flip=True,  
        vertical_flip=False) 

augs_gen.fit(x_train)
#################################
#fig,ax=plt.subplots(5,5)
#fig.set_size_inches(15,15)
#for i in range(5):
#    for j in range (5):
#        l=rn.randint(0,len(Z))
#        ax[i,j].imshow(X[l])
#        ax[i,j].set_title('Grocery: '+Z[l])
#                
#plt.tight_layout()

###############################________________________________________________

base_model = VGG16(include_top=False,
                  input_shape = (imgsize,imgsize,3),
                  weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
    
for layer in base_model.layers:
    print(layer,layer.trainable)

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(25,activation='softmax'))
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

checkpoint = ModelCheckpoint(
    './base.model',
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)
tensorboard = TensorBoard(
    log_dir = './logs',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [checkpoint,tensorboard,csvlogger,reduce]

opt = SGD(lr=1e-4,momentum=0.99)
opt1 = Adam(lr=1e-3)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt1,
    metrics=['accuracy']
)

history = model.fit_generator(
    augs_gen.flow(x_train,y_train,batch_size=128),
    validation_data  = (x_test,y_test),
    validation_steps = 10,                    #100
    steps_per_epoch  = 10,           #100
    epochs = 5,  #50
    verbose = 1,
    callbacks=callbacks
)

#show_final_history(history)
model.load_weights('./base.model')

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    
model.save("model.h5")
print("Weights Saved")


print("_SUCCESS_")






















