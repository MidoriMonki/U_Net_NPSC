import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from data_preparation import prepare_data
from data_preparation import prepare_results
from glob import glob
import nibabel as nib
from sklearn.model_selection import train_test_split
from matplotlib import cm
from keras.models import load_model
import random
import numpy as np

#Get data
images, masks, ct = prepare_data('test_data')

#Load the model
model = load_model('my_model.keras')

for i in range(1):

	pred = []
	ran = random.randint(0, len(images)-1)
	
	for i in range(len(images[ran])):
		test_img = images[ran][i]
		ground_truth=masks[ran][i]
		test_img_norm=test_img[:,:,0][:,:,None]
		test_img_input=np.expand_dims(test_img_norm, 0)
		#pred.append(model.predict(test_img_input))
		pred.append((model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8))
		

	pred = np.array(pred)
	prepare_results(ct[ran], pred)

	plt.figure(figsize=(16, 8))
	
	plt.subplot(231)
	plt.title('PET scan')
	plt.imshow(images[ran][0])

	plt.subplot(232)
	plt.title('Ground Truth')
	plt.imshow(masks[ran][0])

	plt.subplot(233)
	plt.title('Predicted Segmentation')
	plt.imshow(pred[0])


	plt.show()