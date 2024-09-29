import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from glob import glob
import nibabel as nib

from matplotlib import cm

import random
import numpy as np


def prepare_data():
	# load the files
	patient_f = f'data_nii/*/'
	pet_f = glob(patient_f + '/PET.nii.gz')
	pet_f = nib.load(pet_f[0])
	ct_f = glob(patient_f + '/CTres.nii.gz')
	ct_f = nib.load(ct_f[0])
	seg_f = glob(patient_f + '/SEG.nii.gz')
	seg_f = nib.load(seg_f[0])

	#get data from files
	pet = pet_f.get_fdata()
	ct = ct_f.get_fdata()
	seg = seg_f.get_fdata()

	#convert to numpy array
	pet = np.array(pet)
	ct = np.array(ct)
	seg = np.array(seg)

	#swap axes
	pet = np.transpose(pet, axes=[2, 1, 0])
	ct = np.transpose(ct, axes=[2, 1, 0])
	seg = np.transpose(seg, axes=[2, 1, 0])

	#expand dimensions
	pet = np.expand_dims(pet, -1)
	ct = np.expand_dims(ct, -1)
	seg = np.expand_dims(seg, -1)

	#imm = []
	images = []
	masks = []

	for i in range(0, len(pet)):
		#crop our scans
		temp_pet = pet[i][80:320, 80:320]
		temp_ct = ct[i][80:320, 80:320]
		temp_seg = seg[i][80:320, 80:320]

		ok = temp_ct
		cv2.addWeighted(temp_ct, 1, temp_pet, 0.1, 2, ok)

		images.append(ok)
		masks.append(temp_seg)

	return np.array(images), np.array(masks)

#cmap='gray'

#images_gif = np.squeeze(images, axis=3) 
#masks_gif = np.squeeze(masks, axis=3) 

#im = []
#for i in images_gif:
#	im.append(Image.fromarray(i))
#im[0].save('images.gif', save_all=True, append_images=im[1:])

#ma = []
#for i in masks_gif:
#	ma.append(Image.fromarray(i))
#ma[0].save('masks.gif', save_all=True, append_images=ma[1:])

#img, msk = prepare_data()

#for i in range(0, 10):
#
#	x = random.randint(0, len(img)-1)
#
#	plt.figure(figsize=(16, 8))
#	plt.subplot(231)
#	plt.title('Testing Image')
#	plt.imshow(img[x])
#
#	plt.subplot(233)
#	plt.title('mewo')
#	plt.imshow(msk[x])
#
#	plt.show()