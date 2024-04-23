#!/usr/bin/env python
# coding: utf-8

# # Datalist

import numpy as np
import torch
import monai #monai 1.2.0 (python 3.8)
import os.path as osp
import os

monai.utils.misc.set_determinism(seed=218341029) #deterministic training
torch.multiprocessing.set_sharing_strategy('file_system') # it solves https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189

"""
This script loads a dataset from 'baseDir' directory and perform initial
preprocessing operations that must be done for each sample of the dataset.
For example:
    crops
    resampling
    ...

This assumes that both 'image' and 'label' keys must be processed
"""
folds = 5
for fold in range(folds):
######### PARAMETERS #########
    pixdim = [1,1,1]                    # mm # RAS (final pixel dim after resampling) # RAS (towards Right, towards Anterior, towards Superior)
    margin = np.array([30,30,30])       # mm # RAS (add margin to foreground (label) crop)
    baseDir = osp.realpath('balanced_dataset')   # baseDir from which to load the dataset
    datasetFilename = f'dataset_LA_cross_val_fold_{fold}.json' # relative to baseDir
    #write_dir = './prova_odir'          # save in another directory
    write_dir = baseDir                 # save in the same directory as the original dataset
    output_ext='.nii.gz'
    num_workers = 4                     # number of parallel processes to load and process data in parallel
    ##############################


    keys = ['image', 'label']  #keys in dataset_LA.json
    # load image and label data path with correct prefixes. Subsample dictionary to drop other paths
    def subsample_dict(datalist, my_keys=['name', 'image', 'label']):
        return [{key:dict_[key] for key in my_keys} for dict_ in datalist]
    datalist = subsample_dict(monai.data.load_decathlon_datalist(osp.join(baseDir, datasetFilename), data_list_key='training'))
    print(datalist)
    assert len(datalist) > 0
    datalist = datalist
    for i in range(len(datalist)):
        print(i, datalist[i]['name'])

    # this is a function to select points of space only when CA are present at the same z coordinate
    def axially_reduce(array):
        idx_bool = torch.any(torch.any(array>0, dim=-2), dim=-2)
        return idx_bool.expand_as(array)

    # this eliminates any non LA segmentations 
    class selectCAd():
        def __call__(self, data_dict):
            label = data_dict['label']
            idx = label > 1.
            label[idx] = 0.
            data_dict['label'] = label
            return data_dict

    # MONAI TRANSFORMS 
    trans = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys, image_only=False),
        selectCAd(),
        monai.transforms.ToTensord(keys),
        ## monai.transforms.DataStatsD(
        ##     keys,
        ##     prefix='Data',
        ##     data_type=True,
        ##     data_shape=True,
        ##     value_range=True,
        ##     data_value=None,
        ##     additional_info=None,
        ## ),
        monai.transforms.EnsureChannelFirstd(keys),
        monai.transforms.Orientationd(keys, axcodes='RAS'),
        monai.transforms.Spacingd(keys, pixdim=pixdim, mode=['bilinear', 'nearest']),
        monai.transforms.AsDiscreted(['label']),
        monai.transforms.CropForegroundd(
            keys,
            'label',
            select_fn=axially_reduce, # crop only along z direction
            channel_indices=None,
            margin=margin/np.array(pixdim), # margin is in mm
            k_divisible=1,
            mode='minimum',
        ),
        monai.transforms.SaveImaged(
            keys,
            meta_keys=['image_meta_dict', 'label_meta_dict'],
            output_dir=write_dir,
            output_postfix='trans', # scritta che mette nel nome dei file modificati
            output_ext=output_ext,
            resample=False,
            output_dtype=np.float32,
            data_root_dir=baseDir,
            separate_folder=False,
        )
    ])

    # # Dataset transforms
    ds = monai.data.Dataset(
        data=datalist,
        transform=trans,
        )

    loader = monai.data.DataLoader(ds, batch_size=1, shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=False)

    for i, data in enumerate(loader):
        print('from main loop: ', i)
        print(data['name'])

    print('finished')
