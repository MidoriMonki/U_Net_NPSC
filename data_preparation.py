import os
import cv2
import numpy as np
from PIL import Image
from glob import glob
import nibabel as nib
import random

def prepare_data(file_name, dest_name):
	patient_f = f''+str(file_name)+'/*/*/'
	pet_f = glob(patient_f + '/PET.nii.gz')
	seg_f = glob(patient_f + '/SEG.nii.gz')
	ct_f = glob(patient_f + '/CTres.nii.gz')

	#For each patient found
	for i in range(len(pet_f)):
		print('Patient '+str(i+1)+'/'+str(len(pet_f)))
		#Get data from file
		pet = (nib.load(pet_f[i]).get_fdata())
		seg = (nib.load(seg_f[i]).get_fdata())
		ct = (nib.load(ct_f[i]).get_fdata())
		
		pet = np.array(pet)
		seg = np.array(seg)
		ct = np.array(ct)

		#Swap axes of the scans
		pet = np.transpose(pet, axes=[2, 1, 0])
		seg = np.transpose(seg, axes=[2, 1, 0])
		ct = np.transpose(ct, axes=[2, 1, 0])

		#Expand dimensions
		pet = np.expand_dims(pet, -1)
		seg = np.expand_dims(seg, -1)
		ct = np.expand_dims(ct, -1)
			
		pet_arr, seg_arr, ct_arr = [], [], []

		for j in range(len(pet)):
			temp_pet = pet[j][88:312, 88:312]
			temp_seg = seg[j][88:312, 88:312]
			temp_ct = ct[j][88:312, 88:312]
			#Used to remove a majority of the empty masked slices to balance the dataset
			#if cv2.sumElems(temp_seg)[0] > 0: #or random.randint(0, 45) == 1:
			pet_arr.append(temp_pet)
			seg_arr.append(temp_seg)
			ct_arr.append(temp_ct)

		pet_arr = np.array(pet_arr)
		seg_arr = np.array(seg_arr)
		ct_arr = np.array(ct_arr)

		#After applying pre-processing, convert back to Nifti format
		final_pet = nib.Nifti1Image(pet_arr, affine=np.eye(4))
		final_seg = nib.Nifti1Image(seg_arr, affine=np.eye(4))
		final_ct = nib.Nifti1Image(ct_arr, affine=np.eye(4))

		if not os.path.exists(dest_name+"/"+str(i+1)):
			os.makedirs(dest_name+"/"+str(i+1))
		nib.save(final_pet, os.path.join(dest_name+"/"+str(i+1), 'PET.nii.gz'))
		nib.save(final_seg, os.path.join(dest_name+"/"+str(i+1), 'SEG.nii.gz'))
		nib.save(final_ct, os.path.join(dest_name+"/"+str(i+1), 'CTres.nii.gz'))

def prepare_results(ct, seg):
	return_ct = []
	save_ct = []

	ct = np.array(ct)
	seg = np.array(seg)

	#Remove the fourth the axes, as PIL convert to RGBA does not accept it
	#ct = np.squeeze(ct, 3)
	#seg = np.squeeze(seg, 3)

	#Rotate the and swap axes of slices to form full body proportions
	ct = np.transpose(ct, axes=[1, 0, 2])
	ct = np.rot90((ct), 1)
	ct = np.rot90((ct), 1)

	#Repeat for the segmentation slices
	seg = np.transpose(seg, axes=[1, 0, 2])
	seg = np.rot90((seg), 1)
	seg = np.rot90((seg), 1)

	print(len(ct))
	print(seg.shape)
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
		
		#Create a mask that represents all the pixel where the segmentation mask is black [0,0,0] (no tumour)
		mask = np.all(rgb <= [0,0,0], axis = -1)

		#Convert all the pixels that are black into transparent pixels [r,g,b,0]
		temp_seg[mask] = [0,0,0,0]

		#Convert the rest of the pixels into a green colour [50,250,50,255]
		temp_seg[np.logical_not(mask)] = [50,250,50,255]

		temp_seg = Image.fromarray(temp_seg)
		temp_ct = Image.fromarray(temp_ct)

		#Paste the segmentation image ontop of the ct image
		temp_ct.paste(temp_seg, (0,0), mask = temp_seg) 
		temp_ct = np.array(temp_ct)

		save_ct.append(Image.fromarray(temp_ct))
		return_ct.append(temp_ct)

	file_name = 'result_1.gif'
	count = 1

	print(np.array(save_ct).shape)
	print(np.array(return_ct).shape)

	if not os.path.exists('whole-body_seg/'):
		os.makedirs('whole-body_seg/')

	#Find file name that has not been used yet
	while glob('whole-body_seg/' + file_name):
		count += 1
		file_name = 'result_' + str(count) + '.gif'

	#Save all the final images into a gif for presentation
	save_ct[0].save('whole-body_seg/' + file_name, save_all=True, append_images=save_ct[:], optimize=False, loop=0)

	return return_ct


#prepare_data('converted_data')