from testing.test_resnet import *
from testing.make_tsne import *
import os

if __name__ =='__main__':
       batch_size = 406
       train_batch_size = 64
       learning_rate = 0.001
       image_size = 224
       using_dataset = 'OfficeHome'
       entropy_hyperparam = 0.
       leave_one_domain = [3]
       load_path = '/data/hdd1/kdy/Image_classification/saved_model/OfficeHome/EntropyBasedMethod_domainAdaptation_finetuning2/'
    #    load_path = f'/data/hdd1/kdy/Image_classification/saved_model/{using_dataset}/EntropyBasedMethod_domainAdaptation2/'

    #    load_filename = load_path+f'resnet18_lr({learning_rate})_imgsize({image_size})_batchsize({train_batch_size})_Entropy({entropy_hyperparam})_leave_one_domain({leave_one_domain[0]}).pth'
       load_filename = load_path + 'resnet18_lr(0.001)_imgsize(224)_batchsize(64_64)_Entropy(0.0)_leave_one_domain(3)_freeze_False.pth'
       testing_resnet_tsne(load_filename,stop_iter=5,epochs=100,batch_size=train_batch_size,learning_rate=learning_rate,image_size=image_size,
        using_model='resnet18',entropy_hyperparam = entropy_hyperparam,
                    random_seed = 2,use_domain = leave_one_domain, 
                    using_dataset = using_dataset,root_path = '/home/eslab/dataset/OfficeHome/',
                    use_gpu=True,gpu_num=[0,1,2,3])