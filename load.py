from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2 as cv
import matplotlib.pyplot as plt
batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model=Sequential() 
model=load_model('data.h5')
img=cv.imread('3.jpg')
img2=cv.resize(img,(28,28));
gray=cv.cvtColor(img2,cv.COLOR_BGR2GRAY);
gray=255-gray
gray=gray.astype('float32')
gray=gray/255
test=gray.reshape(1,28,28,1)
plt.imshow(test[0,:,:,0])
plt.show()
res=model.predict(test,batch_size=batch_size,verbose=1)
print(res[0])
'''for i in range(20,30):
 plt.imshow(x_test[i])
 plt.show()
 res=model.predict(x_test[i,:,:].reshape(1,28,28,1),batch_size=batch_size,verbose=1)
 print(res[0])'''
