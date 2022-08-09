from train.train_resnet import *
import os

if __name__ =='__main__':
       batch_size_list = [32,64,128,256]
       learning_rate_list = [0.0001]
       image_size_list = [224,448]
       for learning_rate in learning_rate_list:
              for batch_size in batch_size_list:
                     for image_size in image_size_list:
                            save_path = '/data/hdd1/kdy/Image_classification/saved_model/'
                            log_path = '/data/hdd1/kdy/Image_classification/logging/'
                            os.makedirs(save_path,exist_ok=True)
                            os.makedirs(log_path,exist_ok=True)
                            log_filename = log_path+f'resnet18_lr({learning_rate})_imgsize({image_size})_batchsize({batch_size}).txt'
                            save_filename = save_path+f'resnet18_lr({learning_rate})_imgsize({image_size})_batchsize({batch_size}).pth'
                            print(f'log_filename : {log_filename}')
                            print(f'save_filename : {save_filename}')
                            training_resnet(log_filename,save_filename,stop_iter=5,epochs=100,batch_size=batch_size,learning_rate=learning_rate,image_size=image_size,
                                   using_model='resnet18',
                                                 random_seed = 2,use_domain = [0,1,2],leave_one_domain = 3, 
                                                 using_dataset = 'OfficeHome',root_path = '/home/eslab/dataset/OfficeHome/',
                                                 use_gpu=True,gpu_num=[0,1,2,3])