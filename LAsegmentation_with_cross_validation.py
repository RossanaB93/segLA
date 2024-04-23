#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import monai
import ignite
import os.path as osp
import os
import matplotlib.pyplot as plt
import data_with_cross_validation_scheme as data_cv
import shutil

#========================DEVICE SETUP========================
# to correctly use monai.handlers.StatsHandler()
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

## ======================== DATA TRANSFORMS =================================
class clipLabeld():
    def __call__(self, data_dict):
        label = data_dict['label']
        label = np.clip(label, 0, 1)
        data_dict['label'] = label
        return data_dict

def get_xforms(mode='train', keys=("image", "label")):
    xforms = [
        
        monai.transforms.LoadImaged(keys),
        #clipLabeld(),
        monai.transforms.EnsureChannelFirstd(keys),
        monai.transforms.Orientationd(keys, axcodes='RAS'),
        #monai.transforms.Spacingd(keys, pixdim=pixdim, mode=['bilinear', 'nearest']),
        monai.transforms.ScaleIntensityRanged(
            keys[0],
            a_min=-350,
            a_max=800,
            b_min=0,
            b_max=1,
        ),
        monai.transforms.AsDiscreted(keys[1]),
        monai.transforms.EnsureTyped(keys),
        #monai.transforms.ResizeWithPadOrCropd(keys, spatial_size=vol_size_trn)
        monai.transforms.SpatialPadd(keys, spatial_size=vol_size_trn) # performs padding to the data. Spacial_size is the size 
        # of output data after padding. 

        # monai.transforms.RandCropByPosNegLabeld(
        #     keys,
        #     label_key=keys[1],
        #     num_samples=16,
        #     spatial_size=vol_size_trn,
        #     pos=1,
        #     neg=1,
        #     image_key=keys[0],
        #     image_threshold=0,
        # )
    ]
    
    #extended transforms for training
    if mode == 'train':
        xforms.extend(
            [
                #monai.transforms.CropForegroundd(keys, source_key='image'),
                #monai.transforms.RandSpatialCropSamplesd(keys, num_samples=4, roi_size=vol_size_trn, random_center=True, random_size=False),
                
                #data augmentation
                monai.transforms.RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.2, 0.2, 0.2),  # 3 parameters control the transform on 3 dimensions
                    shear_range=(0.05,0.05,0.05),
                    scale_range=(0.15, 0.15, 0.15),
                    mode=("bilinear", "nearest"),
                    #as_tensor_output=False,
                ),#
                
                #gaussian noise
                #monai.transforms.RandGaussianNoised(keys[0], prob=0.15, std=0.05),
                #monai.transforms.RandFlipd(keys, spatial_axis=0, prob=0.5),
                #monai.transforms.RandFlipd(keys, spatial_axis=1, prob=0.5),
                #monai.transforms.RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == 'val':
        dtype = (np.float32, np.uint8)
    if mode == 'infer':
        dtype = (np.float32,)
    xforms.extend(
        [
            monai.transforms.CastToTyped(keys, dtype=dtype), 
            monai.transforms.EnsureTyped(keys)
            ])
    return monai.transforms.Compose(xforms)

## ======================== FUNCTIONS DEFINITION =================================
def get_net():
    UNet_meatdata = dict(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=monai.networks.layers.Norm.BATCH
    )
    return monai.networks.nets.UNet(**UNet_meatdata)

trans_post = monai.transforms.Compose([
    monai.transforms.EnsureTyped(keys=['pred', 'label']),
    monai.transforms.Activationsd(keys=['pred'], softmax=True),
    monai.transforms.AsDiscreted(keys=('pred', 'label'), argmax=(True, False), threshold=0.5),
])

def get_inferer(mode='train'):
    #inferer = monai.inferers.SimpleInferer()
    if mode == 'train':
        sw_batch_size, overlap = 1, 0.25
    elif mode == 'val':
        sw_batch_size, overlap = 8, 0.25
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=vol_size_trn,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode='constant',
        sw_device=device,
        #device='cpu',
    )
    return inferer

device = torch.device("cuda:0")
folds = 5

max_epochs = 500   #IMPORTANT: number of maximum epochs
val_interval = 2
start_saving_model = 100 # inizia a salvare il modello dall'epoca 100 in poi e lo salva ogni 100: nel caso in cui non ci siano grandi differenze, si può anche ridurre il numero di epoche
save_model_every = 100 
best_metric = -1
best_metric_epoch = -1

## ======================== PATH =================================
# IMPORTANTE: cambiare il nome a logs ogni volta che cambio dataset/ parametri e Kfold

for kfold in range(folds):

    dataRoot = osp.realpath('balanced_dataset') #fornisce la posizione reale di dataset nel sistema
    datasetFilename = f'dataset_LA_transf_cross_val_fold_{kfold}.json'
    os.chdir(r'/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset')
    print(osp.join(dataRoot,datasetFilename))
    trialdir = 'RUNS' # cartella in cui vanno i risultati della rete
    logs = f'log_K{kfold}_cross-val_E500_w128'  # sottocartelle di cui devo cambiare il nome ogni volta che runno con parametri diversi o diverso kfold
    logdir = osp.join(trialdir, logs) # crea un percorso valido completo combinando le due stringhe tra parentesi
    #if osp.exists(logdir): shutil.rmtree(logdir)
    if osp.exists(logdir): raise NameError('Output log dir already exists!') # usato per verificare se il percorso esiste già. 
    # Se esiste già, appare il messaggio di errore tra parentesi (raise consente di gestire un errore) -> vogliamo un percorso diverso ogni volta 
    else: os.makedirs(logdir) # viene creata una nuova cartella 
    metric_model = "best_metric_model.pth" # qui vengono salvati i risultati del train
    # kfold to load as validation (the others are loaded as training)

    ## ======================== SETTINGS =================================
    # Window size for inference
    vol_size_trn = [128,128,128] # oppure [64,64,64] o [192,192,192]
    batch_size = 1

    validation_every_n_epochs = 5
    trn_evaluation_every_n_epochs = 10
    init_lr = 1e-4 #initial learning rate (ADAM optimizer -> algoritmo di ottimizzazione alternativo alla discesa del gradiente,
    # usato per aggiornare iterativamente i pesi della rete neurale in base ai dati di addestramento)
    monai.utils.misc.set_determinism(seed=218341029) # imposta il numero massimo di numeri casuali generati durante l'addestramento,
    # così da consentire una certa riproducibilità dei risultati (definita "determinismo")

    ## ======================== DATA LOADING =================================
    # load image and label data path as training and validation dataset
    datalist_trn = data_cv.load_datalist(osp.join(dataRoot,datasetFilename), 'training')
    datalist_val = data_cv.load_datalist(osp.join(dataRoot,datasetFilename), 'validation')
    print(f'NUMBER OF DATASET FOR TRAINING: {len(datalist_trn)}')
    print(datalist_trn) # mostra a video tutti gli indirizzi dei dati usati per il training
    print("-" * 30)
    print(f'NUMBER OF DATASET FOR VALIDATION: {len(datalist_val)}')
    print(datalist_val) # mostra a video tutti gli indirizzi dei dati usati per il validation
    assert len(datalist_trn) > 0
    assert len(datalist_val) > 0


    ## ======================== DATASET LOADER DEFINITION =================================

    ds_trn = monai.data.CacheDataset(
        data=datalist_trn,
        transform=get_xforms(),
        cache_rate=1.0,
        num_workers=4)
    loader_trn = monai.data.DataLoader(
        ds_trn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=monai.data.utils.pad_list_data_collate,
    )

    # ds_trn_eval = monai.data.CacheDataset(
    #     data=datalist_trn,
    #     transform=get_xforms(mode='val'),
    #     cache_rate=1.0,
    #     #cache_num=8,
    #     num_workers=8)
    # loader_trn_eval = monai.data.DataLoader(
    #     ds_trn_eval,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=torch.cuda.is_available(),
    #     collate_fn=monai.data.utils.pad_list_data_collate,
    # )

    ds_val = monai.data.CacheDataset(
        data=datalist_val,
        transform=get_xforms(mode='val'),
        cache_rate=1.0,
        num_workers=1,
    )
    loader_val = monai.data.DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        collate_fn=monai.data.utils.pad_list_data_collate,
    )

    ## ======================== NET DEFINITIONS =================================
    net = get_net().to(device)

    #training loss function: definition
    loss_function = monai.losses.DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    #loss_function = monai.losses.DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
    loss_type = "DiceLoss"
    opt = torch.optim.Adam(net.parameters(), init_lr) # implements Adam algorithm

    #validation metrics: definition
    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean") #average Dice score
    hausdorff_metric = monai.metrics.HausdorffDistanceMetric(include_background=False,reduction='mean')

    #optimizer: definition
    Optimizer_metadata = {}
    for ind, param_group in enumerate(opt.param_groups):
        optim_meta_keys = list(param_group.keys())
        Optimizer_metadata[f'param_group_{ind}'] = {
            key: value for (key, value) in param_group.items() if 'params' not in key
        }

    ## ======================== NET EPOCHS AND SAVINGS =================================

    epoch_loss_values = []
    epoch_loss_idxs = []
    metric_values = []
    h_metric_values = []
    metric_idxs = []
    post_pred = monai.transforms.Compose([monai.transforms.AsDiscrete(argmax=True, to_onehot=2)])
    post_label =monai.transforms.Compose([monai.transforms.AsDiscrete(to_onehot=2)])

    #EPOCHS RUN -> operazioni che effettua per ogni epoca -> calcolo metriche e plotta grafici delle metriche
    for epoch in range(max_epochs): # parte da 0 fino a max_epoch-1
        print("-" * 30) # trenta trattini usati come separatori tra un'epoca e la successiva
        print(f"epoch {epoch + 1}/{max_epochs}") # es. epoch 1/500, epoch 2/500 ecc
        
        #training
        net = net.train()
        epoch_loss = 0
        step = 0
        for batch_data in loader_trn:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            #print(inputs.size())
            opt.zero_grad()
            #outputs = net(inputs)
            roi_size = vol_size_trn
            sw_batch_size = 1
            outputs=monai.inferers.sliding_window_inference(inputs, roi_size, sw_batch_size, net)
            loss = loss_function(outputs, labels) #compute loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(ds_trn) // loader_trn.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        epoch_loss_idxs.append(epoch+1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        np.savetxt(osp.join(logdir, "loss_training.txt"), np.c_[epoch_loss_idxs,epoch_loss_values])
        
        #validation
        if (epoch + 1) % val_interval == 0:
            net = net.eval()
            with torch.no_grad():
                for val_data in loader_val:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = vol_size_trn
                    sw_batch_size = 1
                    val_outputs = monai.inferers.sliding_window_inference(val_inputs, roi_size, sw_batch_size, net)
                    val_outputs = [post_pred(i) for i in monai.data.decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in monai.data.decollate_batch(val_labels)]
                    # compute metric for current iteration

                    dice_metric(y_pred=val_outputs, y=val_labels)
                    print(val_data['name'])

                    #hausdorff_metric(y_pred=val_outputs, y=val_labels)


                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                #h_metric = hausdorff_metric.aggregate().item()
                
                # reset the status for next validation round
                dice_metric.reset()
                #hausdorff_metric.reset()

                metric_values.append(metric)
                #h_metric_values.append(h_metric)
                metric_idxs.append(epoch+1)

                np.savetxt(osp.join(logdir, "dice_metrics_val.txt"), np.c_[metric_idxs,metric_values])
                #np.savetxt(osp.join(logdir, "hausdorff_distance_metrics_val.txt"), np.c_[metric_idxs,h_metric_values])

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(net.state_dict(), osp.join(logdir, metric_model))
                    np.savetxt(osp.join(logdir, "best_epoch.txt"), np.array([best_metric_epoch]))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

                #save no-best-metric model 
                if (epoch+1 > start_saving_model) and ((epoch+1 - start_saving_model)%save_model_every==0):
                    torch.save(net.state_dict(), osp.join(logdir, f"model_{epoch+1:05d}.pth"))


    print(f'NUMBER OF DATASET FOR TRAINING {len(datalist_trn)}')
    print(datalist_trn)
    print("-" * 30)
    print(f'NUMBER OF DATASET FOR VALIDATION {len(datalist_val)}')
    print(datalist_val)
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.ylim(0,1)
    plt.xlim(1,max_epochs)
    plt.plot(x, y)
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = np.ones(len(metric_values))-metric_values
    plt.xlabel("epoch")
    plt.ylim(0,1)
    plt.xlim(1,max_epochs)
    plt.plot(x, y)
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Hausdorff distance")
    x = [val_interval * (i + 1) for i in range(len(h_metric_values))]
    y = h_metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()
    plt.savefig(osp.join(logdir, 'loss_and_dice.png'))

    exit(0)



