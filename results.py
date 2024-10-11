from data_preparation import prepare_results
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import nibabel as nib
from keras.models import load_model
import random
import numpy as np
from keras.metrics import MeanIoU

#Load the model
model = load_model('u_net_model_og.keras')

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

#get data
X_test = get_data_from_dir('greg', 'PET')
y_test = get_data_from_dir('greg', 'SEG')
ct_test = get_data_from_dir('greg', 'CTres')

pred = model.predict(X_test)
truth = y_test

iou = MeanIoU(num_classes=2)
iou.update_state(truth, pred)
print("IoU score of: ", iou.result().numpy())

#For each patient
for i in range(len(X_test)):

	#Predict the segmentation mask
	#test_img_number = random.randint(0, len(X_test))
	#test_img = X_test[test_img_number]
	#ground_truth=y_test[test_img_number]
	#test_img_norm=test_img[:,:,0][:,:,None]
	#test_img_input=np.expand_dims(test_img_norm, 0)
	#pred = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

	#Create the full-body scan
	prepare_results(ct_test[i], pred[i])
     
	plt.figure(figsize=(16, 8))
	plt.subplot(231)
	plt.title('Testing Image')
	plt.imshow(X_test[i])
	plt.subplot(232)
	plt.title('Testing Label')
	plt.imshow(y_test[i], cmap='gray')
	plt.subplot(233)
	plt.title('Prediction on test image')
	plt.imshow(pred[i], cmap='gray')
	plt.show()