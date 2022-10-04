from models.ResNet import *
from utils.function import *
from utils.dataloader_custom import *

from info.OfficeHome_info import*
from info.PACS_info import*
from tqdm import tqdm


import torch
import torchvision.models as models
import torch.nn.functional as F
from models.ResNet import *
from torchsummary import summary
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import time
import sys
import os
import random
import numpy as np
import multiprocessing

class Entropy_each(nn.Module):
    def __init__(self):
        super(Entropy_each, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)

        return b

def training_resnet(load_filename,batch_size=256,image_size=224,entropy_value=0.1,
                    using_model='resnet18',
                               random_seed = 2,use_domain = [0,1,2],leave_one_domain = 3, 
                               using_dataset = 'OfficeHome',root_path = '/home/eslab/dataset/OfficeHome/',
                               use_gpu=True,gpu_num=[0,1,2,3]):
    
    
    if using_dataset == 'OfficeHome':
        class_num = 65
    elif using_dataset == 'PACS':
        class_num = 7

    b1 = 0.5
    b2 = 0.999

    beta = 0.001
    norm_square = 2
    

    cpu_num = multiprocessing.cpu_count()
    cuda = torch.cuda.is_available()
    gpu_num = [0,1,2,3]

    print(f'cpu num  = {cpu_num}')
    print(f'cuda : {cuda}')
    print(f'gpu num : {gpu_num}')
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    random.seed(random_seed) # seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    list_dir = os.listdir(root_path)


    check_dir = []

    for check_name in list_dir:
        if os.path.isdir(root_path + check_name+'/'):
            check_dir.append(root_path + check_name + '/')
        
    print(check_dir)
    
    if using_dataset == 'OfficeHome':
        dataset_dic = dataset_dic_officeHome.copy()
    elif using_dataset == 'PACS':
        dataset_dic = dataset_dic_PACS

    img_list = search(root_path)
    # print(img_list)
    for file_path in img_list:
        class_name = file_path.split('/')[-2]
        domain_name = file_path.split('/')[-3]
        dataset_dic[domain_name][class_name].append(file_path)
    
    train_list,val_list, test_list, leave_list = split_officehome_dataset_leave_one_domain_out(dataset_dic,use_domain,leave_one_domain)

    print(len(train_list),len(val_list),len(test_list),len(leave_list))
    
    if using_dataset == 'OfficeHome':
        # custom dataloader     
        train_dataset = get_officehome_loader(data_list=train_list,image_size=image_size,training=True)
        val_dataset = get_officehome_loader(data_list=val_list,image_size=image_size,training=False)
        test_dataset = get_officehome_loader(data_list=test_list,image_size=image_size,training=False)
    leave_one_dataset = get_officehome_loader(data_list=leave_list,image_size=image_size,training=False)
    # weight & count about training set(for generalization)
    weights,count = make_weights_for_balanced_classes(train_dataset.dataset_list,using_dataset,class_num)
    
    # sampler = control mini-batch samples ratio using weights
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))

    # dataloader (training set, validation set, test set and leave-one-out-domain dataset(All))
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,sampler=sampler,num_workers=(cpu_num//4))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False, num_workers=(cpu_num//4))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False, num_workers=(cpu_num//4))
    leave_one_dataloader = DataLoader(dataset=leave_one_dataset, batch_size=batch_size,shuffle=False, num_workers=(cpu_num//4))
    
    print('='*20 + 'dataloader length' + '='*20)
    print(train_dataset.__len__(),val_dataset.__len__(),test_dataset.__len__(),leave_one_dataset.__len__())
    
    if using_model =='resnet18':    
        model = resnet18(pretrained=False)
    else:
        print('---None model architecture---')
    
    in_features = model.fc.in_features
    # for key,value in model.
    model.fc = nn.Linear(in_features,class_num)
    model.load_state_dict(torch.load(load_filename)['model_state_dict'])
    print('validation : ', torch.load(load_filename)['best_val_acc'])
    summary(model.cuda(),(3,224,224))
    
    if cuda:
        print('can use CUDA!!!')
        model = model.to(device)
        if len(gpu_num) > 1:
            print('Multi GPU Activation !!!', torch.cuda.device_count())
            model = nn.DataParallel(model,device_ids=gpu_num)
        else:
            print('Single GPU Activation !!!')
    else:
        print('Use CPU version')    
    
    # loss function
    loss_fn = Entropy_each().to(device)
    
    
    
    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc = []

    best_val_acc = 0.
    best_epoch = 0
    
    
    

    train_total_loss = 0.0
    train_total_count = 0
    train_total_data = 0
    
    val_total_loss = 0.0
    val_total_count = 0
    val_total_data = 0

    start_time = time.time()
    
    model.eval()

    
    check_entropy_count = 0
    train_total_list = []
    val_total_list = []
    test_total_list = []
    leave_total_list = []
    with tqdm(train_dataloader,desc='Train',unit='batch') as tepoch:
        for index,(img, label,domain) in enumerate(tepoch):
            img = img.to(device)
            label = label.long().to(device)
            with torch.no_grad():
            
                pred = model(img)
                
                loss = loss_fn(pred) # + beta * norm
                
                train_total_list.append(loss.cpu())
                # current_entropy_count = [1 if x < 0.5 for x in loss]

                check_entropy_count += torch.sum(loss < entropy_value)
                _, predict = torch.max(pred, 1)
                check_count = (predict == label).sum().item()

                

                train_total_count += check_count
                train_total_data += len(img)
            accuracy = train_total_count / train_total_data
            tepoch.set_postfix(accuracy=100.*accuracy)
            
    train_total_loss /= index
    train_accuracy = train_total_count / train_total_data * 100

    output_str = 'epochs spend time : %.4f sec / correct : %d/%d -> %.4f%%\n' \
                % (time.time() - start_time,
                    train_total_count, train_total_data, train_accuracy)
    total_train_loss.append(train_total_loss)
    total_train_acc.append(train_accuracy)
    sys.stdout.write(output_str)
    
    print(f'check_entropy_count = {check_entropy_count} // total number of samples = {train_total_data}')

    # check validation dataset
    start_time = time.time()
    model.eval()
    check_entropy_count = 0
    with tqdm(val_dataloader,desc='Validation',unit='batch') as tepoch:
        for index,(img, label,domain) in enumerate(tepoch):
            img = img.to(device)
            label = label.long().to(device)

            with torch.no_grad():
            
                pred = model(img)
                loss = loss_fn(pred) # + beta * norm
                check_entropy_count += torch.sum(loss < entropy_value)
                val_total_list.append(loss.cpu())
                # acc
                _, predict = torch.max(pred, 1)
                check_count = (predict == label).sum().item()

                
                val_total_count += check_count
                val_total_data += len(img)
                accuracy = val_total_count / val_total_data
                tepoch.set_postfix(accuracy=100.*accuracy)


    val_total_loss /= index
    val_accuracy = val_total_count / val_total_data * 100

    output_str = 'epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                % ( time.time() - start_time,
                    val_total_count, val_total_data, val_accuracy)
    sys.stdout.write(output_str)

    print(f'check_entropy_count = {check_entropy_count} // total number of samples = {val_total_data}')
    test_total_count = 0
    test_total_data = 0
    # check validation dataset
    start_time = time.time()
    model.eval()    
    check_entropy_count = 0          
    with tqdm(test_dataloader,desc='Test',unit='batch') as tepoch:
        for index,(img, label,domain) in enumerate(tepoch):
            img = img.to(device)
            label = label.long().to(device)

            with torch.no_grad():
                pred = model(img)

                
                loss = loss_fn(pred)
                check_entropy_count += torch.sum(loss < entropy_value)
                test_total_list.append(loss.cpu())
                # acc
                _, predict = torch.max(pred, 1)
                check_count = (predict == label).sum().item()

                test_total_count += check_count
                test_total_data += len(img)
                accuracy = test_total_count / test_total_data
                tepoch.set_postfix(accuracy=100.*accuracy)


    test_accuracy = test_total_count / test_total_data * 100
    best_test_accuracy = test_accuracy
    output_str = 'epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                % (time.time() - start_time,
                    test_total_count, test_total_data, test_accuracy)
    sys.stdout.write(output_str)
    print(f'check_entropy_count = {check_entropy_count} // total number of samples = {test_total_data}')
    test_total_count = 0
    test_total_data = 0
    check_entropy_count = 0
    with tqdm(leave_one_dataloader,desc='leave_one_dataset',unit='batch') as tepoch:
        for index,(img, label,domain) in enumerate(tepoch):
            img = img.to(device)
            label = label.long().to(device)

            with torch.no_grad():
                pred = model(img)

                loss = loss_fn(pred)
                check_entropy_count += torch.sum(loss < entropy_value)
                leave_total_list.append(loss.cpu())
                # acc
                _, predict = torch.max(pred, 1)
                check_count = (predict == label).sum().item()

                test_total_count += check_count
                test_total_data += len(img)
                accuracy = test_total_count / test_total_data
                tepoch.set_postfix(accuracy=100.*accuracy)


    test_accuracy = test_total_count / test_total_data * 100
    best_leave_test_accuracy = test_accuracy
    output_str = 'epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                % (time.time() - start_time,
                    test_total_count, test_total_data, test_accuracy)
    sys.stdout.write(output_str)
    print(f'check_entropy_count = {check_entropy_count} // total number of samples = {test_total_data}')
    plt.hist(train_total_list)
    plt.savefig('train_total_list.jpg')
    plt.cla()
    plt.clf()

    plt.hist(val_total_list)
    plt.savefig('val_total_list.jpg')
    plt.cla()
    plt.clf()

    plt.hist(test_total_list)
    plt.savefig('test_total_list.jpg')
    plt.cla()
    plt.clf()

    plt.hist(leave_total_list)
    plt.savefig('leave_total_list.jpg')
    plt.cla()
    plt.clf()
if __name__ =='__main__':
    load_filename = '/data/hdd1/kdy/Image_classification/saved_model/resnet18_lr(0.01)_imgsize(224)_batchsize(256).pth'
    training_resnet(load_filename,batch_size=256,image_size=224,
                    using_model='resnet18',
                               random_seed = 2,use_domain = [0,1,2],leave_one_domain = 3, 
                               using_dataset = 'OfficeHome',root_path = '/home/eslab/dataset/OfficeHome/',
                               use_gpu=True,gpu_num=[0,1,2,3])