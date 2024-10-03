from u_net_functions import build_unet
from data_preparation import prepare_input
from dice_loss import dice_loss
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import tifffile as tiff
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam

from glob import glob
from sklearn.model_selection import train_test_split 
import nibabel as nib

import random
import numpy as np

#Load data, if data is already loaded then replace this line to read the saved files
#Data is loaded an a numpy array of all the patient scans, where each index is a different patient
images, masks = prepare_input('training_data')

"""
#For testing purposes, print the shape of the images and masks arrays
for i in images.shape:
    print(i)

for i in masks.shape:
    print(i)
"""

#Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size = 0.25, random_state = 1)

"""
#For testing purposes, display random data to see if the data was successfully set up
n = random.randint(len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (300, 400)), cmap='gray')
plt.imshow(X_train[image_number], cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.imshow(y_train[image_number], cmap='gray')
plt.show()
"""

#The model is designed to take in 240 x 240 x 1 images
shape = (240, 240, 1)

#Build the model
model = build_unet(shape)

#Compile the model
model.compile(optimizer=Adam(learning_rate = 0.001), loss='dice', metrics=['accuracy'])

#Print summary of the model parametres
model.summary()

#Set up call back that completes the model training early if the performance stops increasing
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[callback], batch_size=32, epochs=50)

#Save the model after it has completed training
model.save("my_model.keras")

#Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Dice Score Co-efficient
pred = model.predict(X_test)
dl = dice_loss(y_test, pred, 1)
print("Dice Score Coefficient Loss : ", dl)


#IOU
"""
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)
"""
"""
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()
"""