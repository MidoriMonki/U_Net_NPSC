from u_net_functions import build_unet
from data_preparation import prepare_data
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
from keras import callbacks

from glob import glob
from sklearn.model_selection import train_test_split 
import nibabel as nib

import random
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#prepare_data('200')

def count_data(dirname):
    count = 0
    patient_f = f''+dirname+'/*/'
    pet_f = glob(patient_f + '/PET.nii.gz')
	
	#For each patient found
    for i in range(len(pet_f)):
        pet = (nib.load(pet_f[i]).get_fdata())
        count += len(pet)
        print(i)
    print(count)
    return count

def get_data_from_dir(dirname, filename):
    X = []
    patient_f = f''+dirname+'/*/'
    file_f = glob(patient_f + '/'+filename+'.nii.gz')
	
	#For each patient found
    for i in range(len(file_f)):
        file = (nib.load(file_f[i]).get_fdata())
        print(np.array(file).shape)
        for j in range(len(file)):
            X.append(file[j])
    return np.array(X)


#New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
#This gives a binary mask rather than a mask with interpolated values. 
seed=22

img_data_gen_args = dict(rotation_range=5,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True)

mask_data_gen_args = dict(rotation_range=5,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

batch_size=32

image_data_generator = ImageDataGenerator(None)#**img_data_gen_args)
image_generator = image_data_generator.flow(get_data_from_dir('train_data', 'PET'), seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(get_data_from_dir('validate_data', 'PET'), seed=seed, batch_size=batch_size) #Default batch size 32, if not specified here

mask_data_generator = ImageDataGenerator(None)#**mask_data_gen_args)
mask_generator = mask_data_generator.flow(get_data_from_dir('train_data', 'SEG'), seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(get_data_from_dir('validate_data', 'SEG'), seed=seed, batch_size=batch_size)  #Default batch size 32, if not specified here

def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

my_generator = my_image_mask_generator(image_generator, mask_generator)
validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

"""
x = next(image_generator)
y = next(mask_generator)
for i in range(0,1):
    image = x[0]
    mask = y[0]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()
"""
"""
history = model.fit_generator(my_generator, validation_data=validation_datagen, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, epochs=1)
"""


"""
def generator(filename):
    patient_f = f''+filename+'/*/'
    pet_f = glob(patient_f + '/PET.nii.gz')
    seg_f = glob(patient_f + '/SEG.nii.gz')

    #For each patient found
    for i in range(len(pet_f)):
        #Get data from file
        print(i)
        pet = (nib.load(pet_f[i]).get_fdata())
        seg = (nib.load(seg_f[i]).get_fdata())
        for j in range(len(pet)):
            pet[j] = np.expand_dims( np.array(pet[j]) , 0)
            print(j)
            yield(np.array(pet[j]), np.array(seg[j]))
"""


#For testing purposes, print the shape of the images and masks arrays
#print(X_train.shape)
#print(X_test.shape)

"""
#Split the data into training and testing data
images, masks = [], []

patient_f = f'train_data/*/'
pet_f = glob(patient_f + '/PET.nii.gz')
seg_f = glob(patient_f + '/SEG.nii.gz')

#For each patient found
for i in range(len(pet_f)):
    pet = (nib.load(pet_f[i]).get_fdata())
    seg = (nib.load(seg_f[i]).get_fdata())
    #print(np.array(pet).shape)
    for j in range(len(pet)):
        images.append(pet[j])
        masks.append(seg[j])


X_train, X_test, y_train, y_test = train_test_split(np.array(images), np.array(masks), test_size = 0.25, random_state = 0)
del images
del masks
"""
#For testing purposes, display random data to see if the data was successfully set up
"""
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
#plt.imshow(np.reshape(X_train[image_number], (300, 400)), cmap='gray')
plt.imshow(X_train[image_number], cmap='gray')
plt.subplot(122)
#plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.imshow(y_train[image_number], cmap='gray')
plt.show()
"""
#The model is designed to take in 240 x 240 x 1 images
shape = (224, 224, 1)

#Build the model
model = build_unet(shape)

#Compile the model
model.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy', metrics=['accuracy'])

#Print summary of the model parametres
model.summary()

#Set up call back that completes the model training early if the performance stops increasing
callback = callbacks.EarlyStopping(monitor='val_loss', patience=3)
train_epoch = count_data('train_data')//batch_size
validate_epoch = count_data('validate_data')//batch_size
#Fit the model
history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=train_epoch, validation_steps=validate_epoch, callbacks=[callback], epochs=2)
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, callbacks=[callback], epochs=20)

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


#IOU
"""
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)


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

pred = (model.predict(X_test)[0,:,:,0] > 0.2).astype(np.uint8)

pred = np.array(pred)

plt.figure(figsize=(16, 8))
	
plt.subplot(231)
plt.title('PET scan')
plt.imshow(images[100])

plt.subplot(232)
plt.title('Ground Truth')
plt.imshow(masks[100])

#lt.subplot(233)
#plt.title('Predicted Segmentation')
#plt.imshow(pred)

plt.show()
"""

