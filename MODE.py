

##IMPORTAR LIBRERIAS
import imgaug
import cv2 as cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import matplotlib.image as mpimg
from imgaug  import augmenters as iaa
##REVISAR TENSORFLOW 
print("Versión de tensorflow:{}".format(tf.__version__))
##VER GPU
print("GPU:{}".format(tf.test.gpu_device_name()))
##DIRECCION
path= "DATOS"
##CANTIDAD DE DATOS 
from UTI import *
from MODELONVIDIA import *

data= importDataInfo(path)
##BALANCEAR DATA 
data= BALANCEO_DATOS(data,display=True)

##LEER DATOS 
LeerDatos1(path,data)
Carpetaimagenes, sterring = LeerDatos1(path,data)
##print(Carpetaimagenes[0], sterring[0])

##ENTRENAMIENTO Y VAL DATA 
from sklearn.model_selection import train_test_split
xTrain,xVal, yTrain,yVal =train_test_split(Carpetaimagenes,sterring,test_size=0.3,random_state=4)
print('DATA DE ENTRENAMIENTO',len(xTrain))
print('DATA DE VALIDACION',len(xVal))

## MODELO DE LA RED
model = MODELO_NVIDIA()
model.summary()

##ENTRENAMIENTO
history =model.fit(batchgene(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,validation_data=batchgene(xVal,yVal,100,0),validation_steps=200)

##GRAFICAR
model.save('proto.h5')
print('modelo guardado')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend('ENTRENAMIENTO','VALIDACION')
plt.ylim([0,0.1])
plt.title('loss')
plt.xlabel('EPOCA')
plt.show()