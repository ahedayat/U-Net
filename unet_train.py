import os
import torch
import warnings


import data_loader.camvid as camvid
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils as util

import torchvision.transforms as transforms


def _main(args):
    annalysis_num = args.problem

    num_classes, input_channels, depth, second_layer_channels = (1,3,5,64)
    data_transform = None
    target_transform = None 

    train_data_path, val_data_path, test_data_path = [ './datasets/camvid/{}'.format(data_type) for data_type in ['train','val','test']]
    annotations_version = 2
    
    check_counter = 4 

    loading_model_path, loading_model_name = ('./models','unet')
    saving_model_path, saving_model_name = ('./models','unet')

    report_path = './reports'


    torch.manual_seed(0)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    warnings.filterwarnings("ignore") 

    train_data, val_data, test_data = [ camvid.loader(
                                                data_path, 
                                                anns_dir_name = 'main_ann',
                                                ann_version=annotations_version, 
                                                data_transform=data_transform, 
                                                target_transform=target_transform)
                                        for data_path in [train_data_path, val_data_path, test_data_path]]

    num_classes = train_data.num_labels()
    model = None

    if annalysis_num==2:
        import nets.unet5x5 as unet
    if annalysis_num==4:
        import nets.unet_conv_s2 as unet
    elif annalysis_num==5:
        import nets.unet_BN as unet
    else:
        import nets.unet as unet

    model = unet.Model( num_classes, 
                        input_channels=input_channels, 
                        depth=depth, 
                        second_layer_channels=second_layer_channels )

    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    if args.gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda(device=device)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # nn.CrossEntropyLoss(reduce=False, reduction='sum')
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = unet.EnergyFunction()
    # criterion = nn.MSELoss()
    assert args.optimization in ['adam','sgd'], 'Uknown optimization algorithm. for optimization can use adam and sgd.'
    if args.optimization=='sgd'or annalysis_num==3:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimization=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.load and os.path.isdir(loading_model_path) and os.path.isfile('./{}/{}.pth'.format(loading_model_path, loading_model_name)):
        model, optimizer = unet.load(loading_model_path, loading_model_name, model, optimizer)
    
    if args.start_epoch != 0:
        model_path = './reports/analysis_{}/models'.format(annalysis_num)
        model_name = 'unet_epoch_{}'.format(args.start_epoch-1)
        model, optimizer = unet.load(model_path, model_name, model, optimizer)

    unet.train( model,
                train_data,
                val_data,
                optimizer,
                criterion,
                device,
                num_epoch = args.epochs,
                batch_size = args.batchsize,
                num_workers = args.num_workers,
                check_counter = check_counter,
                gpu=args.gpu,
                saving_model_path='{}/analysis_{}'.format(report_path,annalysis_num),
                run_val=False,
                start_epoch=args.start_epoch)

    if args.save:
        if not os.path.isdir(saving_model_path):
            os.makedirs(saving_model_path)
        unet.save( saving_model_path, saving_model_name, model, optimizer )


if __name__ == '__main__':
    args = util.get_args()
    _main(args)