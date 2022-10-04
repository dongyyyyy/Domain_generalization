from . import *
from sklearn.manifold import TSNE # sklearn 사용하면 easy !! 
from matplotlib import pyplot as plt

def testing_resnet_tsne(load_filename,stop_iter=5,epochs=100,batch_size=256,learning_rate=0.1,image_size=224,
                    using_model='resnet18',
                               random_seed = 2,use_domain = [0,1,2],leave_one_domain = 3, entropy_hyperparam = 0.,
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
    

    # random.seed(random_seed) # seed
    # np.random.seed(random_seed)
    # torch.manual_seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    list_dir = os.listdir(root_path)


    check_dir = []

    for check_name in list_dir:
        if os.path.isdir(root_path + check_name+'/'):
            check_dir.append(root_path + check_name + '/')
        
    print(check_dir)
    
    
    if using_dataset == 'OfficeHome':
        dataset_dic = officeHome_dic()
    elif using_dataset == 'PACS':
        dataset_dic = dataset_dic_PACS

    img_list = search(root_path)
    # print('img_list len = ',len(img_list))
    for file_path in img_list:
        class_name = file_path.split('/')[-2]
        domain_name = file_path.split('/')[-3]
        dataset_dic[domain_name][class_name].append(file_path)

    train_list,val_list, test_list = split_officehome_dataset_one_domain(dataset_dic,use_domain)
    print('list len')
    print(len(train_list),len(val_list),len(test_list))
    train_dataset = None
    val_dataset = None
    test_dataset = None
    leave_one_dataset = None
    if using_dataset == 'OfficeHome':
        # custom dataloader     
        train_dataset = get_officehome_loader(data_list=train_list,image_size=image_size,training=True)
        val_dataset = get_officehome_loader(data_list=val_list,image_size=image_size,training=False)
        test_dataset = get_officehome_loader(data_list=test_list,image_size=image_size,training=False)
        # leave_one_dataset = get_officehome_loader(data_list=leave_list,image_size=image_size,training=False)
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
    print(train_dataset.__len__(),val_dataset.__len__(),test_dataset.__len__())
    # sleep(5)
    # return
    if using_model =='resnet18':    
        model = resnet18(pretrained=False)
    else:
        print('---None model architecture---')
    
    
    in_features = model.fc.in_features
    # for key,value in model.
    
    model.fc = nn.Linear(in_features,class_num)

    # load model weight
    model.load_state_dict(torch.load(load_filename)['model_state_dict'])
    model.fc = nn.Identity()
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
    loss_fn = nn.CrossEntropyLoss().to(device)
    if entropy_hyperparam > 0.:
        print('use Entropy Loss Function')
        loss_entropy = Entropy().to(device)
    # optimization
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=learning_rate, momentum=0.9)
    
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc = []

    best_val_acc = 0.
    best_epoch = 0
    
    output_str = f'model name : {using_model} // use dataset : {using_dataset} // image size : {image_size} // batch size : {batch_size} // init learning rate : {learning_rate}\n'
    sys.stdout.write(output_str)
    
    
    
    for epoch in range(1):
        train_total_loss = 0.0
        train_total_count = 0
        train_total_data = 0
        
        val_total_loss = 0.0
        val_total_count = 0
        val_total_data = 0


        # save model
        
        
        test_total_count = 0
        test_total_data = 0
        # check validation dataset
        start_time = time.time()
        model.eval() 
        deep_features = [] 
        actual = []            
        with tqdm(test_dataloader,desc='Test',unit='batch') as tepoch:
            for index,(img, label,domain) in enumerate(tepoch):
                img = img.to(device)
                label = label.long().to(device)

                with torch.no_grad():
                    pred = model(img)
                    deep_features += pred.cpu().numpy().tolist()
                    actual += label.cpu().numpy().tolist()
                    
        tsne = TSNE(n_components=2, random_state=0) # 사실 easy 함 sklearn 사용하니..
        cluster = np.array(tsne.fit_transform(np.array(deep_features)))
        actual = np.array(actual)

        plt.figure(figsize=(10, 10))
        officehome = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 
        'Calendar' 'Candles' 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 
        'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 
        'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
        'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
        'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 
        'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']
        for i, label in zip(range(65), officehome):
            idx = np.where(actual == i)
            plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)

        # plt.legend()
        plt.savefig('./tsne.png')