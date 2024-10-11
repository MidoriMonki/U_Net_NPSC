## U_Net_NPSC
Below are the steps you would take to get the data and model setup

# Preparing Data
Link to dataset: <a href="https://doi.org/10.7937/gkr0-xv29">

If you would like to use the same dataset from TCIA, be wary of the fact you need a license, and may need to make some changes to files.

First, if you are using MACOS, the OS may generate DS_Store files- which the code used to convert the format does not account for, and will crash.
To fix this, simply change the identify_modalities() function in the tcia_dicom_to_nifti.py file to confirm the given directory is not a DS_Store file.
An example of this change is included in the TCIA folder that contains my modified version of the code.

Second, a few of the python packages were slightly out of date and seemed to course confliction issues with deprecated functions.
One issue I ran into was the read_file() function. I replaced it with pydicom.dcmread() on multiple files on my computer, and that seemed to do the trick.

To actually prepare the data after conversion, put all the converted files into a folder named 'converted_data'.
Then run data_preparation.py- this will prepare all the data and put it into a folder named 'prepared_data'.

Next, put your data into 3 folders 'train_data', 'validate_data', and 'test_data'

# Running Model

Run u_net.py- this will take the data from our 'train_data' and 'validate_data', and will save copy of the model name 'u_net_model.keras'

# Results

After u_net.py has been run and there is a 'u_net_model.keras' file, run results.py- this will take the data from our 'test_data', and uses it to evaluate
the model with and IOU score, and develop full-body PET/CT segmentation scans of each patient. 