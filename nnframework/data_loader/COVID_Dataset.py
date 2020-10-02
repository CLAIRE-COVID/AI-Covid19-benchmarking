__copyright__ = "Isaak Kavasidis"
__email__ = "ikavasidis@gmail.com"

import torch
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import pydicom
import csv,json
import random
from shutil import copyfile
import nibabel as nib
from torchvision import transforms as T
import re
from PIL import Image

class COVID_Split(Dataset):

    def __init__(self, files, labels, sublabels, transform = None, replicate_channel=False, self_supervised=False, random_rotations=True):
        '''
        Args:
        - self_supervised: if True, apply random rotations and return label corresponding to applied rotation
        - random_rotations: if False, apply rotation deterministically (based on item index)
        '''

        super(COVID_Split, self).__init__()
        self.files = files
        self.labels = labels
        self.sublabels = sublabels
        self.self_supervised = self_supervised
        self.random_rotations = random_rotations
        self.replicate_channel = replicate_channel
        
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
        
        #img = img - img.min()
        #img = img / img.max()

        # Get labels
        label = self.labels[idx]
        sublabel = self.sublabels[idx]

        # Check self train
        if self.self_supervised:
            # Compute label
            label = random.randint(0, 3)
            sublabel = 0
            # Apply rotation (expected size: CxHxW)
            img = img.rot90(dims=(1,2), k=label)
        # Check channel replication
        if self.replicate_channel:
            img = img.expand(3, -1, -1)
        
        if torch.isnan(img).any():
            print("NAN")
        return img, label, sublabel
    

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

                    label = (0 if self.covid not in lab[0] else 1)
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

            label = (0 if self.covid not in lab[0] else 1)
            sub_label = ([0 if self.ground_glass not in lab[0] else 1],[0 if self.consolidation not in lab[0] else 1])

            [labels.append(label) for i in range(len(files))]
            [sublabels.append(sub_label) for i in range(len(files))] 
            return files_old,labels,sublabels
    
    
        #mode: possible values "ct" and "xray"
        #plane (not used): define the plane of acquisition
        #splits: a list containing decimal numbers that represent the percentages of the various splits. If len(splits) == 2 or len(splits) == 3 and splits[1] == 0 it omits the validations split
        #root: the path to the generated dataset 
        #pos_neg_file: the absolute path to the file labels_covid19_posi.tsv
    
    def __init__(self, root, mode = "xray", plane = "axial", splits = [0.6,0.2,0.2], replicate_channel=False, batch_size=2, input_size=224, num_workers=2, pos_neg_file = None, random_seed = None):
        
        self.ground_glass = "C3544344"
        self.consolidation = "C0521530"
        self.covid = "C5203670"
        self.replicate_channel = replicate_channel
        
        self.has_val = (len(splits) == 3 and splits[1] != 0)
        self.subjects = os.listdir(root)
        
        if random_seed == None:
            random.shuffle(self.subjects)
        else:
            random.seed(random_seed)
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
            
        train_transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ])
        test_transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
        ])
        
        # Dataset instances
        self.train = COVID_Split(self.train_files,self.train_labels,self.train_sublabels,train_transform,replicate_channel=self.replicate_channel)
        self.val = COVID_Split(self.val_files,self.val_labels,self.val_sublabels,test_transform, replicate_channel=self.replicate_channel)
        self.test = COVID_Split(self.test_files,self.test_labels,self.test_sublabels,test_transform, replicate_channel=self.replicate_channel)         

        # Loader instances
        self.train = DataLoader(self.train, batch_size=batch_size, num_workers=num_workers)
        self.val = DataLoader(self.val, batch_size=batch_size, num_workers=num_workers)
        self.test = DataLoader(self.test, batch_size=batch_size, num_workers=num_workers)
