
# coding: utf-8
#Udacity Self Driving Car - Nanodegree
#Author- Subhendu Mishra
# In[1]:

import csv
import cv2
import numpy as np


# In[2]:
#Open the driving_log file to read the training data
lines = []
with open('/input/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# In[3]:
#Images are the input images from the camera
#Measurement are the steering angles
#For each steering angle, three camera images are taken as the input
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = '/input/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


# In[4]:
#Data Augmentation
#The input images are flipped to increase the amount of data
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


# In[5]:
#X_train correspond to the input camera images
#y_train correspond to the steering angle
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# In[6]:
#Import keras = 2.0.4
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout


# In[7]:
#Function for resizing and normalization of each image
def resize_normalize(image):
    import cv2
    from keras.backend import tf as ktf   
    resized = ktf.image.resize_images(image, (64, 64))
    #normalize 0-1
    resized = resized/255.0 - 0.5
    return resized


# In[8]:

#Model Architecture
model = Sequential()

#Layer 1 has the cropped image removing the part which dows not include the road
model.add(Cropping2D(cropping=((60,20), (1,1)), input_shape=(160,320,3)))

#Layer 2 adds the normalized and resized plane 3@64x64
model.add(Lambda(resize_normalize, input_shape=(160, 320, 3), output_shape=(64, 64, 3)))

#Layer 3 Convolutional Feature Map input = (64,64,3) output = (30,30,24)
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

#Layer 4 Convolutional Feature Map input = (30,30,24) output = (13,13,36)
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

#Layer 5 Convolutional Feature Map input = (13,13,36) output = (5,5,48)
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

#Dropout is not recommeded as it deviates the car out of tack
#model.add(Dropout(0.3))

#Layer 6 Convolutional Feature Map input = (5,5,36) output = (3,3,64)
model.add(Convolution2D(64,3,3,activation="relu"))

#Layer 7 Convolutional Feature Map input = (3,3,36) output = (1,1,64)
model.add(Convolution2D(64,3,3,activation="relu"))


model.add(Flatten())

#Layer 8 Fully connected layer
model.add(Dense(100))

#Layer 9 Fully connected layer
model.add(Dense(50))

#Layer 10 Fully connected layer
model.add(Dense(10))

#Output: Vehicle Control
model.add(Dense(1))


# In[9]:

#Compile with optimizer = 'Adam'
model.compile(loss='mse', optimizer='adam')

#Fit the model with 20% data as validation 
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs = 7)


# In[10]:
#Save the model as model.h5
model.save('model.h5')




