import numpy as np
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms
# from info.OfficeHome_info import *
import os 
from PIL import Image
import random

from info.OfficeHome_info import *
from info.PACS_info import *

def make_weights_for_balanced_classes(data_list, using_dataset,nclasses=65):
    count = [0] * nclasses
    if using_dataset == 'OfficeHome':
        for data in data_list:
            count[class_dic_officeHome[data.split('/')[-2]]] += 1
    elif using_dataset == 'PACS':
        for data in data_list:
            count[class_dic_PACS[data.split('/')[-2]]] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(data_list)

    if using_dataset == 'OfficeHome':
        for idx, val in enumerate(data_list):
            weight[idx] = weight_per_class[class_dic_officeHome[val.split('/')[-2]]]
    elif using_dataset == 'PACS':
        for idx, val in enumerate(data_list):
            weight[idx] = weight_per_class[class_dic_PACS[val.split('/')[-2]]]

    return weight , count

class get_officehome_loader(object):
    def __init__(self,data_list,
                           image_size=224,
                           training=False):

        self.dataset_list = data_list
        self.domain_dic = { 0: 'Art' ,  1 :'Clipart' ,  2 : 'Product', 3 : 'RealWorld'}
        self.class_dic={'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 
        'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 
        'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 
        'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 
        'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 
        'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 
        'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
        self.domain_dic = dict([(value, key) for key, value in self.domain_dic.items()])
        print(f'img size = {image_size}')
        if training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((int(image_size),int(image_size)),scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                # transforms.Resize((int(image_size),int(image_size))),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # normalize,
            ])
            # 0.08 ~ 1 면적의 비율을 우선 자름(Crop) -> ratio 만큼으로 너비와 높이의 비율을 0.75~1.33333으로 조절 -> 최종적으로 size크기로 Resize
        else:
            self.transform = transforms.Compose([
                transforms.Resize((int(image_size),int(image_size))),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                # normalize,
            ])
        
    def __getitem__(self,index):
        image = Image.open(self.dataset_list[index]).convert('RGB') # BGR --> RGB
        label = int(self.class_dic[self.dataset_list[index].split('/')[-2]])
        domain = int(self.domain_dic[self.dataset_list[index].split('/')[-3]])
        return self.transform(image), label, domain

    def __len__(self):
        return len(self.dataset_list)


class get_PACS_loader(object):
    def __init__(self,data_list,
                           image_size=224,
                           training=False):

        self.dataset_list = data_list
        self.domain_dic = { 0: 'art_painting' ,  1 :'cartoon' ,  2 : 'photo', 3 : 'sketch'}
        self.class_dic= {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}
        self.domain_dic = dict([(value, key) for key, value in self.domain_dic.items()])
        print(f'img size = {image_size}')
        if training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((int(image_size),int(image_size)),scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                # transforms.Resize(int(image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # normalize,
            ])
            # 0.08 ~ 1 면적의 비율을 우선 자름(Crop) -> ratio 만큼으로 너비와 높이의 비율을 0.75~1.33333으로 조절 -> 최종적으로 size크기로 Resize
        else:
            self.transform = transforms.Compose([
                transforms.Resize((int(image_size),int(image_size))),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
                # normalize,
            ])
        
    def __getitem__(self,index):
        image = Image.open(self.dataset_list[index]).convert('RGB')
        label = int(self.class_dic[self.dataset_list[index].split('/')[-2]])
        domain = int(self.domain_dic[self.dataset_list[index].split('/')[-3]])
        return self.transform(image), label, domain

    def __len__(self):
        return len(self.dataset_list)

def split_officehome_dataset_oneDomain(dataset_path='/home/eslab/dataset/OfficeHome/RealWorld/',train_split=0.8,val_split=0.1,test_split=0.1):
    train_files = []
    val_files = []
    test_files = []
    dataset_list = os.listdir(dataset_path)
    number_of_classes = len(dataset_list)
    for dataset_folder in dataset_list:
        image_path = dataset_path+dataset_folder+'/'
        image_list = os.listdir(image_path)
        random.shuffle(image_list)
        length = len(image_list)
        train_length = int(length*train_split)
        val_length = int(length*val_split)
        test_length = length - train_length - val_length
        index = 0
        for image_filename in image_list:
            if image_filename.split('.')[-1] == 'jpg' or image_filename.split('.')[-1] == 'png':
                image_filename = image_path + image_filename
                if train_length > index:
                    train_files.append(image_filename)
                elif train_length <= index and (train_length+val_length) > index:
                    val_files.append(image_filename)
                else:
                    test_files.append(image_filename)
                index += 1
        # print(index)
    return train_files,val_files,test_files,number_of_classes

# k-fold split
def split_officehome_dataset_leave_one_domain_out(dataset_dic,use_domain,leave_one_domain_index,train_split=0.8,val_split=0.1,test_split=0.1):    
    
    domain_dic = { 0: 'Art' ,  1 :'Clipart' ,  2 : 'Product', 3 : 'RealWorld'}
    class_dic={'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 
        'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 
        'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 
        'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 
        'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 
        'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 
        'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
    class_dic = dict([(value, key) for key, value in class_dic.items()])
    train_files = {0:[],1:[],2:[],3:[]}
    val_files = {0:[],1:[],2:[],3:[]}
    test_files = {0:[],1:[],2:[],3:[]}
    # print(domain_dic[leave_one_domain_index])
    leave_one_domain_set = dataset_dic[domain_dic[leave_one_domain_index]]
    leave_one_domain_files = []
    
    for class_index in range(len(list(class_dic.keys()))):
        leave_one_domain_files += leave_one_domain_set[class_dic[class_index]]
        
    # save_domain_index = 0
    domain_len = {0:0,1:0,2:0,3:0}
    for indexing,domain_key in enumerate(dataset_dic):
        for class_key in dataset_dic[domain_dic[0]]:
            domain_len[indexing] += len(dataset_dic[domain_key][class_key])
    print('='*30)
    print('Total Domain dataset length')
    print(domain_len[0],domain_len[1],domain_len[2],domain_len[3])
    
    for domain_index in range(len(list(domain_dic.keys()))):
        #if domain_index != leave_one_domain_index:
        for class_index in range(len(list(class_dic.keys()))):
            total_len = len(dataset_dic[domain_dic[domain_index]][class_dic[class_index]])
            train_len = int(total_len * train_split)
            val_len = int(total_len * val_split)
            test_len = int(total_len * test_split)
            val_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][:val_len]
            test_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][val_len:test_len+val_len]
            train_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][test_len+val_len:]
            # save_domain_index += 1
    print('='*30)
    print('train_files')            
    print(len(train_files[0]),len(train_files[1]),len(train_files[2]),len(train_files[3]))
    
    print('='*30)
    print('val_files')            
    print(len(val_files[0]),len(val_files[1]),len(val_files[2]),len(val_files[3]))
    print('='*30)
    
    print('test_files')            
    print(len(test_files[0]),len(test_files[1]),len(test_files[2]),len(test_files[3]))
    print('='*30)
    print('leave-one-domain-files length')
    print(len(leave_one_domain_files))
    print('='*30)

    train_list = []
    val_list = []
    test_list = []
    for i in range(len(train_files)):
        if i in use_domain:
                train_list += train_files[i]
                val_list  += val_files[i]
                test_list += test_files[i]

    
    if test_split == 0.:
        return train_list, val_list, _, leave_one_domain_files
        # return train_files, val_files, leave_one_domain_files
    else:
        return train_list, val_list, test_list, leave_one_domain_files
    # class_num = len(list(dataset_dic[list(dataset_dic.keys())[0]].keys()))
    # print('num of class : ',class_num)
    
    # return train_files, val_files,test_files, leave_one_domain_files

def split_officehome_dataset_one_domain(dataset_dic,use_domain,train_split=0.8,val_split=0.1,test_split=0.1):    
    
    domain_dic = { 0: 'Art' ,  1 :'Clipart' ,  2 : 'Product', 3 : 'RealWorld'}
    class_dic={'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 
        'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 
        'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 
        'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 
        'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 
        'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 
        'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}
    class_dic = dict([(value, key) for key, value in class_dic.items()])
    train_files = {0:[],1:[],2:[],3:[]}
    val_files = {0:[],1:[],2:[],3:[]}
    test_files = {0:[],1:[],2:[],3:[]}
    # print(domain_dic[leave_one_domain_index])
    
        
    # save_domain_index = 0
    domain_len = {0:0,1:0,2:0,3:0}
    for indexing,domain_key in enumerate(dataset_dic):
        for class_key in dataset_dic[domain_dic[0]]:
            domain_len[indexing] += len(dataset_dic[domain_key][class_key])
    print('='*30)
    print('Total Domain dataset length')
    print(domain_len[0],domain_len[1],domain_len[2],domain_len[3])
    
    for domain_index in range(len(list(domain_dic.keys()))):
        #if domain_index != leave_one_domain_index:
        for class_index in range(len(list(class_dic.keys()))):
            total_len = len(dataset_dic[domain_dic[domain_index]][class_dic[class_index]])
            train_len = int(total_len * train_split)
            val_len = int(total_len * val_split)
            test_len = int(total_len * test_split)
            val_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][:val_len]
            test_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][val_len:test_len+val_len]
            train_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][test_len+val_len:]
            # save_domain_index += 1
    print('='*30)
    print('train_files')            
    print(len(train_files[0]),len(train_files[1]),len(train_files[2]),len(train_files[3]))
    
    print('='*30)
    print('val_files')            
    print(len(val_files[0]),len(val_files[1]),len(val_files[2]),len(val_files[3]))
    print('='*30)
    
    print('test_files')            
    print(len(test_files[0]),len(test_files[1]),len(test_files[2]),len(test_files[3]))
    print('='*30)
    

    train_list = []
    val_list = []
    test_list = []
    for i in range(len(train_files)):
        if i in use_domain:
                train_list += train_files[i]
                val_list  += val_files[i]
                test_list += test_files[i]

    return train_list, val_list, test_list
    # class_num = len(list(dataset_dic[list(dataset_dic.keys())[0]].keys()))
    # print('num of class : ',class_num)
    
    # return train_files, val_files,test_files, leave_one_domain_files

def split_PACS_dataset_leave_one_domain_out(dataset_dic,use_domain,leave_one_domain_index,train_split=0.8,val_split=0.1,test_split=0.1):    
    
    domain_dic = { 0: 'art_painting' ,  1 :'cartoon' ,  2 : 'photo', 3 : 'sketch'}
    class_dic={'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}
    class_dic = dict([(value, key) for key, value in class_dic.items()])
    train_files = {0:[],1:[],2:[],3:[]}
    val_files = {0:[],1:[],2:[],3:[]}
    test_files = {0:[],1:[],2:[],3:[]}
    # print(domain_dic[leave_one_domain_index])
    leave_one_domain_set = dataset_dic[domain_dic[leave_one_domain_index]]
    leave_one_domain_files = []
    
    for class_index in range(len(list(class_dic.keys()))):
        leave_one_domain_files += leave_one_domain_set[class_dic[class_index]]
        
    # save_domain_index = 0
    domain_len = {0:0,1:0,2:0,3:0}
    for indexing,domain_key in enumerate(dataset_dic):
        for class_key in dataset_dic[domain_dic[0]]:
            domain_len[indexing] += len(dataset_dic[domain_key][class_key])
    print('='*30)
    print('Total Domain dataset length')
    print(domain_len[0],domain_len[1],domain_len[2],domain_len[3])
    
    for domain_index in range(len(list(domain_dic.keys()))):
        #if domain_index != leave_one_domain_index:
        for class_index in range(len(list(class_dic.keys()))):
            total_len = len(dataset_dic[domain_dic[domain_index]][class_dic[class_index]])
            train_len = int(total_len * train_split)
            val_len = int(total_len * val_split)
            test_len = int(total_len * test_split)
            val_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][:val_len]
            test_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][val_len:test_len+val_len]
            train_files[domain_index] += dataset_dic[domain_dic[domain_index]][class_dic[class_index]][test_len+val_len:]
            # save_domain_index += 1
    print('='*30)
    print('train_files')            
    print(len(train_files[0]),len(train_files[1]),len(train_files[2]),len(train_files[3]))
    
    print('='*30)
    print('val_files')            
    print(len(val_files[0]),len(val_files[1]),len(val_files[2]),len(val_files[3]))
    print('='*30)
    
    print('test_files')            
    print(len(test_files[0]),len(test_files[1]),len(test_files[2]),len(test_files[3]))
    print('='*30)
    print('leave-one-domain-files length')
    print(len(leave_one_domain_files))
    print('='*30)

    train_list = []
    val_list = []
    test_list = []
    for i in range(len(train_files)):
        if i in use_domain:
                train_list += train_files[i]
                val_list  += val_files[i]
                test_list += test_files[i]

    
    if test_split == 0.:
        return train_list, val_list, _, leave_one_domain_files
        # return train_files, val_files, leave_one_domain_files
    else:
        return train_list, val_list, test_list, leave_one_domain_files