from . import *

def training_resnet(log_file,save_file,stop_iter=5,epochs=100,batch_size=256,learning_rate=0.1,image_size=224,
                    using_model='resnet18',
                               random_seed = 2,use_domain = [0,1,2],leave_one_domain = 3, 
                               using_dataset = 'OfficeHome',root_path = '/home/eslab/dataset/OfficeHome/',
                               use_gpu=True,gpu_num=[0,1,2,3]):
    
    
    print('logging filename : ',log_file)
    print('save filename : ',save_file)
    check_file = open(log_file, 'w')
    
    if using_dataset == 'OfficeHome':
        class_num = 65

    b1 = 0.5
    b2 = 0.999

    beta = 0.001
    norm_square = 2
    
    image_size = 224

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
    weights,count = make_weights_for_balanced_classes(train_dataset.dataset_list)
    
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
        model = resnet18(pretrained=True)
    else:
        print('---None model architecture---')
    
    in_features = model.fc.in_features
    # for key,value in model.
    model.fc = nn.Linear(in_features,class_num)
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
    check_file.write(output_str)
    
    
    for epoch in range(epochs):
        train_total_loss = 0.0
        train_total_count = 0
        train_total_data = 0
        
        val_total_loss = 0.0
        val_total_count = 0
        val_total_data = 0

        start_time = time.time()
        
        model.train()

        output_str = 'classification_lr : %f\n'%(optimizer.state_dict()['param_groups'][0]['lr'])
        sys.stdout.write(output_str)
        check_file.write(output_str)
        
        with tqdm(train_dataloader,desc='Train',unit='batch') as tepoch:
            for index,(img, label,domain) in enumerate(tepoch):
                img = img.to(device)
                label = label.long().to(device)

                optimizer.zero_grad()
                
                pred = model(img)

                loss = loss_fn(pred, label) # + beta * norm

                _, predict = torch.max(pred, 1)
                check_count = (predict == label).sum().item()

                train_total_loss += loss.item()

                train_total_count += check_count
                train_total_data += len(img)
                loss.backward()
                optimizer.step()
                accuracy = train_total_count / train_total_data
                tepoch.set_postfix(loss=train_total_loss/(index+1),accuracy=100.*accuracy)
                
        # scheduler.step(epoch)
        scheduler.step()
        
        train_total_loss /= index
        train_accuracy = train_total_count / train_total_data * 100

        output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                    % (epoch + 1, epochs, time.time() - start_time, train_total_loss,
                        train_total_count, train_total_data, train_accuracy)
        total_train_loss.append(train_total_loss)
        total_train_acc.append(train_accuracy)
        sys.stdout.write(output_str)
        check_file.write(output_str)
        

        # check validation dataset
        start_time = time.time()
        model.eval()

        with tqdm(val_dataloader,desc='Validation',unit='batch') as tepoch:
            for index,(img, label,domain) in enumerate(tepoch):
                img = img.to(device)
                label = label.long().to(device)

                with torch.no_grad():
                
                    pred = model(img)
                    loss = loss_fn(pred, label) # + beta * norm


                    # acc
                    _, predict = torch.max(pred, 1)
                    check_count = (predict == label).sum().item()

                    val_total_loss += loss.item()
                    val_total_count += check_count
                    val_total_data += len(img)
                    accuracy = val_total_count / val_total_data
                    tepoch.set_postfix(loss=val_total_loss/(index+1),accuracy=100.*accuracy)


        val_total_loss /= index
        val_accuracy = val_total_count / val_total_data * 100

        output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                    % (epoch + 1, epochs, time.time() - start_time, val_total_loss,
                        val_total_count, val_total_data, val_accuracy)
        sys.stdout.write(output_str)
        check_file.write(output_str)
        total_val_loss.append(val_total_loss)
        total_val_acc.append(val_accuracy)

        # save model
        
        if epoch == 0:
            best_val_acc = val_accuracy
            best_epoch = epoch
            if len(gpu_num) > 1:
                torch.save({'model_state_dict':model.module.state_dict(),
                            'epoch' : epoch,
                            'optimizer_sate_dict':optimizer.state_dict(),
                            'best_val_acc':best_val_acc,
                            'learning_rate':optimizer.state_dict()['param_groups'][0]['lr'],
                            'scheduler':scheduler}, save_file)
            else:
                torch.save({'model_state_dict':model.state_dict(),
                            'epoch' : epoch,
                            'optimizer_sate_dict':optimizer.state_dict(),
                            'best_val_acc':best_val_acc,
                            'learning_rate':optimizer.state_dict()['param_groups'][0]['lr'],
                            'scheduler':scheduler}, save_file)
            stop_count = 0
            stop_count = 0
            test_total_count = 0
            test_total_data = 0
            # check validation dataset
            start_time = time.time()
            model.eval()              
            with tqdm(test_dataloader,desc='Test',unit='batch') as tepoch:
                for index,(img, label,domain) in enumerate(tepoch):
                    img = img.to(device)
                    label = label.long().to(device)

                    with torch.no_grad():
                        pred = model(img)

                        loss = loss_fn(pred, label)

                        # acc
                        _, predict = torch.max(pred, 1)
                        check_count = (predict == label).sum().item()

                        test_total_count += check_count
                        test_total_data += len(img)
                        accuracy = test_total_count / test_total_data
                        tepoch.set_postfix(accuracy=100.*accuracy)


            test_accuracy = test_total_count / test_total_data * 100
            best_test_accuracy = test_accuracy
            output_str = 'test dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                        % (epoch + 1, epochs, time.time() - start_time,
                            test_total_count, test_total_data, test_accuracy)
            sys.stdout.write(output_str)
            check_file.write(output_str)

            with tqdm(leave_one_dataloader,desc='leave_one_dataset',unit='batch') as tepoch:
                for index,(img, label,domain) in enumerate(tepoch):
                    img = img.to(device)
                    label = label.long().to(device)

                    with torch.no_grad():
                        pred = model(img)

                        loss = loss_fn(pred, label)

                        # acc
                        _, predict = torch.max(pred, 1)
                        check_count = (predict == label).sum().item()

                        test_total_count += check_count
                        test_total_data += len(img)
                        accuracy = test_total_count / test_total_data
                        tepoch.set_postfix(accuracy=100.*accuracy)


            test_accuracy = test_total_count / test_total_data * 100
            best_leave_test_accuracy = test_accuracy
            output_str = 'leave test dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                        % (epoch + 1, epochs, time.time() - start_time,
                            test_total_count, test_total_data, test_accuracy)
            sys.stdout.write(output_str)
            check_file.write(output_str)
        else:
            if best_val_acc < val_accuracy:
                best_val_acc = val_accuracy
                best_epoch = epoch
                
                if len(gpu_num) > 1:
                    torch.save({'model_state_dict':model.module.state_dict(),
                            'epoch' : epoch,
                            'optimizer_sate_dict':optimizer.state_dict(),
                            'best_val_acc':best_val_acc,
                            'learning_rate':optimizer.state_dict()['param_groups'][0]['lr'],
                            'scheduler':scheduler}, save_file)
                else:
                    torch.save({'model_state_dict':model.state_dict(),
                                'epoch' : epoch,
                                'optimizer_sate_dict':optimizer.state_dict(),
                                'best_val_acc':best_val_acc,
                                'learning_rate':optimizer.state_dict()['param_groups'][0]['lr'],
                                'scheduler':scheduler}, save_file)
                stop_count = 0
                test_total_count = 0
                test_total_data = 0
                # check validation dataset
                start_time = time.time()
                model.eval()              
                with tqdm(test_dataloader,desc='Test',unit='batch') as tepoch:
                    for index,(img, label,domain) in enumerate(tepoch):
                        img = img.to(device)
                        label = label.long().to(device)

                        with torch.no_grad():
                            pred = model(img)

                            loss = loss_fn(pred, label)

                            # acc
                            _, predict = torch.max(pred, 1)
                            check_count = (predict == label).sum().item()

                            test_total_count += check_count
                            test_total_data += len(img)
                            accuracy = test_total_count / test_total_data
                            tepoch.set_postfix(accuracy=100.*accuracy)


                test_accuracy = test_total_count / test_total_data * 100
                best_test_accuracy = test_accuracy
                output_str = 'test dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                            % (epoch + 1, epochs, time.time() - start_time,
                                test_total_count, test_total_data, test_accuracy)
                sys.stdout.write(output_str)
                check_file.write(output_str)

                with tqdm(leave_one_dataloader,desc='leave_one_dataset',unit='batch') as tepoch:
                    for index,(img, label,domain) in enumerate(tepoch):
                        img = img.to(device)
                        label = label.long().to(device)

                        with torch.no_grad():
                            pred = model(img)

                            loss = loss_fn(pred, label)

                            # acc
                            _, predict = torch.max(pred, 1)
                            check_count = (predict == label).sum().item()

                            test_total_count += check_count
                            test_total_data += len(img)
                            accuracy = test_total_count / test_total_data
                            tepoch.set_postfix(accuracy=100.*accuracy)


                test_accuracy = test_total_count / test_total_data * 100
                best_leave_test_accuracy = test_accuracy
                output_str = 'leave test dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                            % (epoch + 1, epochs, time.time() - start_time,
                                test_total_count, test_total_data, test_accuracy)
                sys.stdout.write(output_str)
                check_file.write(output_str)
            else:
                stop_count += 1
                
        if stop_count > stop_iter:
            print('Early Stopping')
            output_str = 'best epoch : %d/%d spend time : %.4f sec  / best test accuracy : %.4f%% / best leave-one-domain accuracy : %.4f%% \n' \
                            % (best_epoch+1, epochs, time.time() - start_time,
                                best_test_accuracy,best_leave_test_accuracy)
            sys.stdout.write(output_str)
            check_file.write(output_str)
            
            output_str = '====End System====\n'
            sys.stdout.write(output_str)
            check_file.write(output_str)
            break