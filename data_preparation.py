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

def prepare_data(file_name):
	patient_f = f''+str(file_name)+'/*/'
	pet_f = glob(patient_f + '/PET.nii.gz')
	ct_f = glob(patient_f + '/CTres.nii.gz')
	seg_f = glob(patient_f + '/SEG.nii.gz')

	pet, ct, seg = [], [], []

	#For each patient found
	for i in range(len(pet_f)):
		#Get data from file
		pet.append(nib.load(pet_f[i]).get_fdata())
		ct.append(nib.load(ct_f[i]).get_fdata())
		seg.append(nib.load(seg_f[i]).get_fdata())
		#Swap axes of the scans
		pet[i] = np.transpose(pet[i], axes=[2, 1, 0])
		ct[i] = np.transpose(ct[i], axes=[2, 1, 0])
		seg[i] = np.transpose(seg[i], axes=[2, 1, 0])
		#Expand dimensions
		pet[i] = np.expand_dims(pet[i], -1)
		ct[i] = np.expand_dims(ct[i], -1)
		seg[i] = np.expand_dims(seg[i], -1)
	
	images = []
	masks = []
	ct_arr = []

	for i in range(len(pet)):
		images.append([])
		ct_arr.append([])
		masks.append([])
		for j in range(len(pet[i])):
			#crop our scans
			temp_pet = pet[i][j][80:320, 80:320]
			temp_ct = ct[i][j][80:320, 80:320]
			temp_seg = seg[i][j][80:320, 80:320]
			#temp_seg = normalize_2d(temp_seg)
			#temp_ct = normalize_2d(temp_ct)
			images[i].append(temp_pet)
			masks[i].append(temp_seg)
			ct_arr[i].append(temp_ct)
			#ok = temp_ct
			#cv2.addWeighted(temp_ct, 1, temp_pet, 1, 1, ok)

	return images, masks, ct_arr



def prepare_input(file_name):
	im, ms, ct = prepare_data(file_name)
	images, masks = [], []

	for i in range(len(im)):
		for j in range(len(im[i])):
			images.append(im[i][j])
			masks.append(im[i][j])
	#Returns data in a numpy array format, so the model can use it
	return np.array(images), np.array(masks)



def prepare_results(ct, seg):
	return_ct = []
	save_ct = []

	ct = np.array(ct)
	seg = np.array(seg)

	#Remove the fourth the axes, as PIL convert to RGBA does not accept it
	ct = np.squeeze(ct, 3)
	#seg = np.squeeze(seg, 3)

	#Rotate the and swap axes of scans to form full body proportions
	ct = np.transpose(ct, axes=[1, 0, 2])
	ct = np.rot90((ct), 1)
	ct = np.rot90((ct), 1)
	#Repeat for the segmentation
	seg = np.transpose(seg, axes=[1, 0, 2])
	seg = np.rot90((seg), 1)
	seg = np.rot90((seg), 1)


	#For each image in the ct scan
	
	for i in range(len(ct)):
		#Convert ct to RGBA image to add colour and alpha channels
		pil_seg = Image.fromarray((seg[i])*255)
		pil_seg = pil_seg.convert('RGBA')
		temp_seg = np.array(pil_seg)
		#Repeat for the segmentation images
		pil_ct = Image.fromarray((ct[i]))
		pil_ct = pil_ct.convert('RGBA')
		temp_ct = np.array(pil_ct)
	
		#Get colour of the segmentation images
		rgb = temp_seg[:,:,:3]
		
		#Create a mask that represents all the pixel where the segmentation mask is black [0,0,0]
		#These pixels are the ones where the mask did not identify any tumours
		mask = np.all(rgb <= [0,0,0], axis = -1)

		#Convert all the pixels that are black into transparent pixels [r,g,b,0]
		temp_seg[mask] = [0,0,0,0]

		#Convert the rest of the pixels into a green [50,250,50,255]
		temp_seg[np.logical_not(mask)] = [50,250,50,255]

		#Convert the scans back to PIL images
		temp_seg = Image.fromarray(temp_seg)
		temp_ct = Image.fromarray(temp_ct)

		#Paste the segmentation image ontop of the ct image
		temp_ct.paste(temp_seg, (0,0), mask = temp_seg) 
		temp_ct = np.array(temp_ct)

		#Append the results to our two arrays
		save_ct.append(Image.fromarray(temp_ct))
		return_ct.append(temp_ct)

	file_name = 'example1.gif'
	count = 1

	if not os.path.exists('whole-body_seg/'):
		os.makedirs('whole-body_seg/')

	#Find file name that has not been used yet
	while glob('whole-body_seg/' + file_name):
		count += 1
		file_name = 'example' + str(count) + '.gif'

	#Save all the final images into a gif for presentation
	save_ct[0].save('whole-body_seg/' + file_name, save_all=True, append_images=save_ct[1:], optimize=False, duration=40, loop=0)

	return return_ct
