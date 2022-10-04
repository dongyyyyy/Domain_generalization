from train.train_resnet import *
from train.train_resnet_withMixStyle import *
from train.train_resnet_finetuning import *
import os

if __name__ =='__main__':
       batch_size_list = [64]
       train_batch_size_list = [64]
       learning_rate_list = [0.001]
       image_size_list = [224]
       using_dataset = 'OfficeHome'
       entropy_hyperparam_list = [0.]
       freeze_list = [True,False]
       leave_one_domain = [3]
       for entropy_hyperparam in entropy_hyperparam_list:
              for learning_rate in learning_rate_list:
                     for batch_size in batch_size_list:
                            for image_size in image_size_list:
                                   for freeze in freeze_list:
                                          for train_batch_size in train_batch_size_list:
                                                 load_path = f'/data/hdd1/kdy/Image_classification/saved_model/{using_dataset}/EntropyBasedMethod_domainAdaptation/'
                                                 save_path = f'/data/hdd1/kdy/Image_classification/saved_model/{using_dataset}/EntropyBasedMethod_domainAdaptation_finetuning2/'
                                                 log_path = f'/data/hdd1/kdy/Image_classification/logging/{using_dataset}/EntropyBasedMethod_domainAdaptation_finetuning2/'
                                                 os.makedirs(save_path,exist_ok=True)
                                                 os.makedirs(log_path,exist_ok=True)
                                                 load_filename = load_path+f'resnet18_lr({learning_rate})_imgsize({image_size})_batchsize({batch_size})_Entropy({entropy_hyperparam})_leave_one_domain({leave_one_domain[0]}).pth'
                                                 log_filename = log_path+f'resnet18_lr({learning_rate})_imgsize({image_size})_batchsize({batch_size}_{train_batch_size})_Entropy({entropy_hyperparam})_leave_one_domain({leave_one_domain[0]})_freeze_{freeze}.txt'
                                                 save_filename = save_path+f'resnet18_lr({learning_rate})_imgsize({image_size})_batchsize({batch_size}_{train_batch_size})_Entropy({entropy_hyperparam})_leave_one_domain({leave_one_domain[0]})_freeze_{freeze}.pth'
                                                 print(f'log_filename : {log_filename}')
                                                 print(f'save_filename : {save_filename}')
                                                 training_resnet_finetuning(load_filename,log_filename,save_filename,freeze=freeze,stop_iter=5,epochs=100,batch_size=train_batch_size,learning_rate=learning_rate,image_size=image_size,
                                                        using_model='resnet18',entropy_hyperparam = entropy_hyperparam,
                                                                      random_seed = 2,use_domain = leave_one_domain, 
                                                                      using_dataset = using_dataset,root_path = '/home/eslab/dataset/OfficeHome/',
                                                                      use_gpu=True,gpu_num=[0,1,2,3])