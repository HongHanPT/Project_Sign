#pip install tensorflow-gpu==2.0.rc ( 2.2)
#pip install wget
#link = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
#wget.download(link)

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import numpy as np
import cv2 as cv
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split as split_data
import time

data = "./"  #Path to data unzip

train_link = data + "train.p"    # Bi dong goi bang pickle
valid_link = data + "valid.p"
test_link = data + "test.p"

# with open(train_link, mode="rb") as f:
#   train = pickle.load(f)
#
# with open(valid_link, mode="rb") as f:
#   valid = pickle.load(f)
#
# with open(test_link, mode="rb") as f:
#   test = pickle.load(f)
#
# # cv.imwrite()
#
# trainX = train["features"]
# trainY = train["labels"]
#
# plt.imshow(trainX[0])
# plt.imshow(trainX[1])
# plt.imshow(trainX[10])
# plt.imshow(trainX[50])

data_path = 'D:\TPA\Projects\Vision\ClassifySign\Data32x32\\'
def create_train_data():
    training_data= []
    for directory in ['Fast', 'Slow','Start','Stop']:
        image_path = os.path.join(data_path, directory)
        if directory == 'Fast':
            label = 0
        elif directory == 'Slow':
            label = 1
        elif directory == 'Start':
            label = 2
        elif directory == 'Stop':
            label =3
        for img in tqdm(os.listdir(image_path)):
            path = os.path.join(image_path, img)
            img_data=cv.imread(path)
            training_data.append([np.array(img_data), label])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

classNames = {0: 'Speed limit (20km/h)',
 1: 'Speed limit (30km/h)',
 2: 'Speed limit (50km/h)',
 3: 'Speed limit (60km/h)',
 4: 'Speed limit (70km/h)',
 5: 'Speed limit (80km/h)',
 6: 'End of speed limit (80km/h)',
 7: 'Speed limit (100km/h)',
 8: 'Speed limit (120km/h)',
 9: 'No passing',
 10: 'No passing for vehicles over 3.5 metric tons',
 11: 'Right-of-way at the next intersection',
 12: 'Priority road',
 13: 'Yield',
 14: 'Stop',
 15: 'No vehicles',
 16: 'Vehicles over 3.5 metric tons prohibited',
 17: 'No entry',
 18: 'General caution',
 19: 'Dangerous curve to the left',
 20: 'Dangerous curve to the right',
 21: 'Double curve',
 22: 'Bumpy road',
 23: 'Slippery road',
 24: 'Road narrows on the right',
 25: 'Road work',
 26: 'Traffic signals',
 27: 'Pedestrians',
 28: 'Children crossing',
 29: 'Bicycles crossing',
 30: 'Beware of ice/snow',
 31: 'Wild animals crossing',
 32: 'End of all speed and passing limits',
 33: 'Turn right ahead',
 34: 'Turn left ahead',
 35: 'Ahead only',
 36: 'Go straight or right',
 37: 'Go straight or left',
 38: 'Keep right',
 39: 'Keep left',
 40: 'Roundabout mandatory',
 41: 'End of no passing',
 42: 'End of no passing by vehicles over 3.5 metric tons'}


# # Shuffle data
# trainX, trainY = shuffle(trainX, trainY)
# #plt.imshow(trainX[0])
#
# validX = valid["features"]
# validY = valid["labels"]
#
# testX = test["features"]
# testY = test["labels"]

# trainX = trainX.astype("float") / 255.0
# validX = validX.astype("float") / 255.0
# testX = testX.astype("float") / 255.0
#
#
# lb = LabelBinarizer()
# trainY = lb.fit_transform(trainY)
# validY = lb.fit_transform(validY)

def ClassifySignModel():
    model = Sequential()
    width = 32
    height = 32
    classes = 4
    shape = (width, height, 3)
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    learning_rate = 0.01
    opt = SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return  model
model = ClassifySignModel()
model.summary()
aug = ImageDataGenerator(rotation_range=0.18, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

dataX= create_train_data()

train, test = split_data(dataX, test_size=0.2, random_state=42)
x_train = np.array([i[0] for i in train])
y_train = [i[1] for i in train]
x_test = np.array([i[0] for i in test])
y_test = [i[1] for i in test]

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
x_train = x_train.astype("float") / 255.0
x_test = x_test.astype("float") / 255.0
print(dataX)

epochs = 10
batch_size = 16
MODEL_NAME = 'SignClassify'
print("Start training")
#model.fit(x_train, y_train, epochs=10,
#          validation_data =(x_test, y_test),batch_size=16,
#          steps_per_epoch=x_train.shape[0]/batch_size)
H = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0]/batch_size, epochs=epochs, verbose=1)
#model.save_weights("classifySign.h5")
# Load weights
saved_model = ClassifySignModel()
saved_model.load_weights("classifySign.h5")
from PIL import Image
img = cv.imread('D:/TPA/Projects/Vision/ClassifySign/0a.jpg')
img = img.reshape((-1,32,32,3))
print(x_test[0].shape)
print(img.shape)
# cv.imshow('',img)
time.sleep(1)
timeq = time.time()
#result = saved_model.predict(x_test[0:10])
result = saved_model.predict(img.astype(float)/255.0)  #---- OK, chua Ok do ham reshape 
time = time.time()
#
print(result, time- timeq)
#
# final = np.argmax(result)
#
# final = classNames[final]
#
# plt.imshow(test["features"][100])


#if __name__ == '__main__':
