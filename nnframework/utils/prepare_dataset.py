__copyright__ = "Isaak Kavasidis"
__email__ = "ikavasidis@gmail.com"

import torch
import os
from os.path import join
from torch.utils.data import Dataset
import numpy as np
import glob
import pydicom
import csv,json
import random
from shutil import copyfile
import nibabel as nib
from torchvision import transforms
import re

###The path to the V1.0 folder of the bimcv dataset
root = r"W:\corona\Bimcv\V1.0"

###The output path where the converted dataset will be stored 
output_folder = r"H:\bimcv_claire\ct\coronal"

def nifty2numpy(nifti_path):
    img = nib.load(nifti_path)
    return np.array(img.dataobj), img.header 


rx_folders = []
tac_folders = []

ground_glass = "C3544344"
consolidation = "C0521530"

folders = os.listdir(root)

for patient in folders:
    
    patient_path = os.path.join(root,patient)
    if os.path.isdir(patient_path):
        found = False
        
        sessions = os.listdir(patient_path)
        for session in sessions:
            session_path = os.path.join(patient_path,session,"mod-rx")
            if os.path.isdir(session_path):     
                
                files = os.listdir(session_path)
                for file in files: 
                    if ".png" in file:
                        os.makedirs(os.path.join(output_folder,'xray',patient),exist_ok=True)
                        copyfile(os.path.join(session_path,file),os.path.join(output_folder,'xray',patient,file))
                    elif '.nii.gz' in file:
                        try:
                            ct_scan,header = nifty2numpy(os.path.join(session_path,file))
                            #This subroutine should work if the dicom tag describing the plane of acquisition was compile, but it wasn't
                            #json_file = file.replace('.nii.gz','.json')
                            
                            #with open(os.path.join(session_path,f)) as json_file:
                                #data = json.load(json_file)
                                
                                #plane = data['00200037']['Value']
                                
                                
                                #if plane[4] == 0 and plane[5] ==  -1:
                                    #plane = 'saggital'
                                #elif plane[4] == 1 and plane[5] == 0:
                                    #plane = 'axial'
                                #else:
                                    #plane = 'coronal'
                               
                            ct_scan = ct_scan - ct_scan.min()
                            ct_scan = ct_scan/ct_scan.max()*255
                            img_cnt = ct_scan.shape[2]
                            ct_result_path = os.path.join(output_folder,'ct'patient,file)
                            os.makedirs(ct_result_path,exist_ok=True)

                            for image in range(img_cnt):
                                img = ct_scan[:,:,image]
                                img = Image.fromarray(np.rot90(img))
                                img = img.convert("L")
                                img.save(os.path.join(ct_result_path, str(image) + ".png" ))
                        except:
                            print(patient + " , " + session)
        