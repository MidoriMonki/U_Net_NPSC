from build_u_net import build_model
from data_preparation import prepare_data
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.optimizers import Adam
from keras import callbacks
from glob import glob
from sklearn.model_selection import train_test_split 
import nibabel as nib
import random

prepare_data('200', 'greg')
images, masks = [], []

patient_dir = f'train_data/*/'
pet_f = glob(patient_dir + '/PET.nii.gz')
seg_f = glob(patient_dir + '/SEG.nii.gz')

#For each patient found
for i in range(len(pet_f)):
    #Retrieve each slice
    pet = (nib.load(pet_f[i]).get_fdata())
    seg = (nib.load(seg_f[i]).get_fdata())
    for j in range(len(pet)):
        images.append(pet[j])
        masks.append(seg[j])

#Split data into training and validation/test
X_train, X_test, y_train, y_test = train_test_split(np.array(images), np.array(masks), test_size = 0.25, random_state = 0)
del images
del masks

#For testing purposes, display some of the slices to see if it was successfully set up
for i in range(20):
    image_number = random.randint(0, len(X_train))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(X_train[image_number])
    plt.subplot(122)
    plt.imshow(y_train[image_number])
    plt.show()

#The model is designed to take in 224 x 224 x 1 images
shape = (224, 224, 1)

#Build the model
model = keras.models.load_model('u_net_model.keras')
#model = build_model(shape)

#Compile the model
model.compile(optimizer=Adam(learning_rate = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Set up call back that completes the model training early if the performance stops increasing
#callback = callbacks.EarlyStopping(monitor='val_loss', patience=3)
#callbacks=[callback]
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10)

#Save the model after it has completed training
model.save("u_net_model.keras")

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
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Find the IOU score using the validation/test data
pred=model.predict(X_test)
pred_thresholded = pred > 0.5

intersection = np.logical_and(y_test, pred_thresholded)
union = np.logical_or(y_test, pred_thresholded)
iou = np.sum(intersection) / np.sum(union)
print("IoU score is: ", ioy)

