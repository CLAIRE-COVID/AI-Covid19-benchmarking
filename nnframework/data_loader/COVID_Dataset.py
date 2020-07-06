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


class COVID_Split(Dataset):

    def __init__(self, files, labels, sublabels, transform = None):
        super(COVID_Split, self).__init__()
        self.files = files
        self.labels = labels
        self.sublabels = sublabels
        
        if transform == None:
            self.transform = transforms.ToTensor()
        else:    
            self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        img = Image.open(self.files[idx])
        
        if self.transform is not None:
            img = self.transform(img)    
        
        img = img - img.min()
        img = img / img.max()
        
        return img.unsqueeze(0) , self.labels[idx], self.sublabels[idx]
    

class COVID_Dataset():
    
    
    
    def addFiles(self,subj,mode,split,files_old,labels,sublabels,root):       
        
        if mode == 'ct':
            runs = os.listdir(os.path.join(root,subj))
            for run in runs:
                run_root = os.path.join(root,subj,run)
                if os.path.isdir(run_root):
                    files = glob.glob(os.path.join(run_root,"*.png"))

                    if len(files)>0:
                        x = re.findall("ses-E[0][0-9][0-9][0-9][0-9]", files[0])
                        session = x[0]
                    else:
                        continue

                    try:
                        lab = self.labels[subj][session]

                    except:
                        continue

                    files_old.extend(files)

                    label = [0 if self.covid not in lab[0] else 1]
                    sub_label = ([0 if self.ground_glass not in lab[0] else 1],[0 if self.consolidation not in lab[0] else 1])

                    [labels.append(label) for i in range(len(files))]
                    [sublabels.append(sub_label) for i in range(len(files))] 
            return files_old,labels,sublabels
        else:
            files = glob.glob(os.path.join(root,subj,"*.png"))
            
            if len(files)>0:
                x = re.findall("ses-E[0][0-9][0-9][0-9][0-9]", files[0])
                session = x[0]
            else:
                return files_old,labels,sublabels

            try:
                lab = self.labels[subj][session]

            except:
                return files_old,labels,sublabels

            files_old.extend(files)

            label = [0 if self.covid not in lab[0] else 1]
            sub_label = ([0 if self.ground_glass not in lab[0] else 1],[0 if self.consolidation not in lab[0] else 1])

            [labels.append(label) for i in range(len(files))]
            [sublabels.append(sub_label) for i in range(len(files))] 
            return files_old,labels,sublabels
    
    
        #mode: possible values "ct" and "xray"
        #plane (not used): define the plane of acquisition
        #splits: a list containing decimal numbers that represent the percentages of the various splits. If len(splits) == 2 or len(splits) == 3 and splits[1] == 0 it omits the validations split
        #root: the path to the generated dataset 
        #pos_neg_file: the absolute path to the file labels_covid19_posi.tsv
    
    def __init__(self, root, mode = "xray", plane = "axial", splits = [0.6,0.2,0.2], transform = None, pos_neg_file = None):
        
        self.ground_glass = "C3544344"
        self.consolidation = "C0521530"
        self.covid = "C5203670"
        
        self.has_val = (len(splits) == 3 and splits[1] != 0)
        self.subjects = os.listdir(root)
        random.shuffle(self.subjects)
        
        self.labels = {}
        
        with open(pos_neg_file) as pos_neg_file:
            skip_first = True
            labels_rdr = csv.reader(pos_neg_file, delimiter="\t")
            
            for row in labels_rdr:
                if skip_first:
                    skip_first = False
                    continue
                
                if row[1] not in self.labels.keys():
                    self.labels[row[1]] = {}
                if row[2] not in self.labels[row[1]].keys():
                    self.labels[row[1]][row[2]] = []
                
                self.labels[row[1]][row[2]].append(row[7].replace("[","").replace("]","").split("\t"))
                
        tot_cnt = len(self.subjects)
        
        self.splits = {}
        self.splits["train"] = self.subjects[0:int(splits[0]*tot_cnt)]
        
        
        if self.has_val == False:
            self.splits["test"] = self.subjects[len(self.splits["train"]):]
        else:
            self.splits["val"] = self.subjects[len(self.splits["train"]):int((splits[0]+splits[1])*tot_cnt)]
            self.splits["test"] = self.subjects[len(self.splits["val"])+len(self.splits["train"]):]
        
        self.train_files = []
        self.test_files = []
        self.val_files = None
        
        self.train_labels = []
        self.test_labels = []
        self.val_labels = None
        
        self.train_sublabels = []
        self.test_sublabels = []
        self.val_sublabels = None
        
        if self.has_val:
            
            self.val_files = [] 
            self.val_labels = []
            self.val_sublabels = []
            for s in self.splits["val"]:
                self.val_files,self.val_labels,self.val_sublabels = self.addFiles(s,mode,"val",self.val_files,self.val_labels,self.val_sublabels,root)
            
        for s in self.splits["train"]:
            self.train_files,self.train_labels,self.train_sublabels = self.addFiles(s,mode,"train",self.train_files,self.train_labels,self.train_sublabels,root)
            
        for s in self.splits["test"]:
            self.test_files,self.test_labels,self.test_sublabels = self.addFiles(s,mode,"test",self.test_files,self.test_labels,self.test_sublabels,root)
            
        
        self.train = COVID_Split(self.train_files,self.train_labels,self.train_sublabels,transform)
        self.val = COVID_Split(self.val_files,self.val_labels,self.val_sublabels,transform)
        self.test = COVID_Split(self.test_files,self.test_labels,self.test_sublabels,transform)         
        
