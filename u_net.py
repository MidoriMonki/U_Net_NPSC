# https://youtu.be/GAYJ81M58y8

from u_net_functions import build_unet
from data_preparation import prepare_data
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
import nibabel as nib

#get data
images, masks = prepare_data()

for i in images.shape:
    print(i)

for i in masks.shape:
    print(i)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size = 0.25, random_state = 0)


#Sanity check, view few images
import random
import numpy as np
image_number = random.randint(0, len(X_train))

#plt.figure(figsize=(12, 6))
#plt.subplot(121)
#plt.imshow(np.reshape(X_train[image_number], (300, 400)), cmap='gray')
#plt.imshow(X_train[image_number], cmap='gray')
#plt.subplot(122)
#plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
#plt.imshow(y_train[image_number], cmap='gray')
#plt.show()




#IMG_HEIGHT = images.shape[1]
#IMG_WIDTH  = images.shape[2]
#IMG_CHANNELS = images.shape[3]

IMG_HEIGHT = 240
IMG_WIDTH  = 240
IMG_CHANNELS = 1

for i in X_train[0].shape:
    print(i)

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_unet(input_shape)
# lr was replaced with learning_rate
model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



#steps_per_epoch = 3*(len(X_train))//batch_size
steps_per_epoch = 2*len(X_train)


#inputt = np.array(list((X_train, y_train)))

#used to be  model.fit_generator
# would be my_generator instead of X_train
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
#                    steps_per_epoch=steps_per_epoch, 
#                    validation_steps=steps_per_epoch, epochs=1)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=25)

#plot the training and validation accuracy and loss at each epoch
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

model.save("my_model.keras")

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

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