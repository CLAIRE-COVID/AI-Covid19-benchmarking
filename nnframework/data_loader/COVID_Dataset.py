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
        
        self.subject_ids = []
        self.slice_ids = []
        self.ct_ids = []
        for f in files:
            parts = os.path.basename(f).split('_')
            self.subject_ids.append(parts[0])
            self.ct_ids.append(parts[1]) 
            self.slice_ids.append(int(parts[-1].split('.')[0]))
     
        if transform == None:
            self.transform = T.ToTensor()
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
        return img, label, sublabel, self.subject_ids[idx], self.ct_ids[idx], self.slice_ids[idx]
    

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

                    # skip subjects with uncertain label
                    if self.uncertain in lab[0]:
                        print('{}-{} is uncertain: skipped'.format(subj,run))
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
    
    def __init__(self, root, mode = "xray", plane = "axial", splits = [0.6,0.2,0.2], replicate_channel=False, batch_size=2, input_size=224, num_workers=2, pos_neg_file = None, random_seed = None, self_supervised=False,test_patients = None):
        
        self.ground_glass = "C3544344"
        self.consolidation = "C0521530"
        self.covid = "C5203670"
        self.uncertain = "C5203671"
        self.replicate_channel = replicate_channel
        
        self.has_val = (len(splits) == 3 and splits[1] != 0)
        self.subjects = os.listdir(root)
        

            
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
                
        
        positive_subjs = []
        negative_subjs = []
        self.splits = {}
        
        if test_patients is not None:
            self.subjects = list( set(self.subjects) - set(test_patients))
            

        for subj in self.subjects:
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

                    # skip subjects with uncertain label
                    if self.uncertain in lab[0]:
                        print('{}-{} is uncertain: skipped'.format(subj,run))
                        continue
                    

                    label = (0 if self.covid not in lab[0] else 1)
                    print('{}: {}'.format(subj,label))
                    if label ==1:
                        positive_subjs.append(subj)
                    else:
                        negative_subjs.append(subj)
                    break


        tot_cnt = len(positive_subjs) + len(negative_subjs)
        pos_cnt = len(positive_subjs)
        neg_cnt = len(negative_subjs)
        if random_seed == None:
            random.shuffle(positive_subjs)
            random.shuffle(negative_subjs)
        else:
            random.seed(random_seed)
            random.shuffle(positive_subjs)
            random.shuffle(negative_subjs)

        

 

        n_neg_test = int(0.20*neg_cnt)
        n_pos_test = n_neg_test
        n_neg_val = int(0.10*neg_cnt)
        n_pos_val = n_neg_val

        pos_test = positive_subjs[0:n_pos_test]
        neg_test = negative_subjs[0:n_neg_test]

        pos_val = positive_subjs[n_pos_test: n_pos_test+n_pos_val]
        neg_val = negative_subjs[n_neg_test: n_neg_test+n_neg_val]

        pos_train = positive_subjs[n_pos_test+n_pos_val:]
        neg_train = negative_subjs[n_neg_test+n_neg_val:]

        if test_patients is not None:
            self.splits["train"] = pos_train + neg_train + pos_test + neg_test
            self.splits["val"] = pos_val + neg_val
            self.splits['test'] = list(set(test_patients))
        else:
                
            self.splits["train"] = pos_train + neg_train
            self.splits["val"] = pos_val + neg_val
            self.splits["test"] = pos_test+neg_test

        # assert patients belong just to one split
        set_test = set(self.splits['test'])
        set_train = set(self.splits['train'])
        set_val = set(self.splits['val'])
        assert len(set_train.intersection(set_test)) == 0, 'Subjects of training set are present in Test set'
        assert len(set_train.intersection(set_val)) == 0, 'Subjects of training set are present in Validation set'
        assert len(set_val.intersection(set_test)) == 0, 'Subjects of validation set are present in Test set'


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
        self.train = COVID_Split(self.train_files,self.train_labels,self.train_sublabels,train_transform,replicate_channel=self.replicate_channel, self_supervised=self_supervised)
        self.val = COVID_Split(self.val_files,self.val_labels,self.val_sublabels,test_transform, replicate_channel=self.replicate_channel, self_supervised=self_supervised)
        self.test = COVID_Split(self.test_files,self.test_labels,self.test_sublabels,test_transform, replicate_channel=self.replicate_channel, self_supervised=self_supervised)         
        
        # Loader instances
        self.train = DataLoader(self.train, batch_size=batch_size, num_workers=num_workers,shuffle=True)
        self.val = DataLoader(self.val, batch_size=batch_size, num_workers=num_workers)
        self.test = DataLoader(self.test, batch_size=batch_size, num_workers=num_workers)

    def get_label_proportions(self, n_classes=2):
        labs = []
        m = 0
        for i in range(n_classes):
            c = self.train_labels.count(i)
            if c > m:
                m = c
            labs.append(c)
        for i in range(n_classes):
            labs[i] = 1/(labs[i] / m)
        return torch.Tensor(labs)
