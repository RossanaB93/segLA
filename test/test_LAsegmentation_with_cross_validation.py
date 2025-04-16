import os
import glob
import pyvista as pv
import numpy as np
import torch
import os.path as osp
import matplotlib.pyplot as plt
import shutil
import monai
# to correctly use monai.handlers.StatsHandler()
import data
import vtk
import nibabel as nb

os.chdir(r"/home/your_working_directory/") 
kfold = 3
## ======================== Datalist =================================
dataRoot = osp.realpath('/home/your_dataset')
datasetFilename = f'dataset_LA_cross_val_fold_{kfold}.json'

# # IMPORTANTE !! : cambiare nome logs e logs_segm
TRIAL_ROOT = osp.realpath('/home/your_dataset_path/RUNS')
logs= f'log_K{kfold}_cross-val_E500_w128'   #folder of the .pth file
logdir = osp.join(TRIAL_ROOT, logs)
metric_model = 'best_metric_model.pth'  #name of the .pth filep
state_dict_path=os.path.join(logdir, metric_model)
SEGM_ROOT = osp.realpath('/home/your_dataset_path')
logs_segm= f'test_K{kfold}_E500' #output segmentation folder
out_name_vtp = 'vtp_output_res'
out_name_stl = 'stl_output_res'
out_name_nii = 'nii_output_res'


## ======================== Functions =================================
def marching_cubes(array):
    array = array.squeeze()
    assert len(array.shape)==3
    a = pv.wrap(array)
    contour = a.contour(
            isosurfaces=1,
            rng=(0.5,1),
            method='marching_cubes',
    )
    return contour

def windowedSincSmooth(mesh, iters=20, passband=0.01):
    smoothed = vtk.vtkWindowedSincPolyDataFilter()
    smoothed.SetInputData(mesh)
    smoothed.SetNumberOfIterations(iters)
    smoothed.SetPassBand(passband)
    smoothed.SetBoundarySmoothing(False)
    smoothed.SetFeatureEdgeSmoothing(False)
    smoothed.SetNonManifoldSmoothing(True)
    smoothed.SetNormalizeCoordinates(True)
    smoothed.Update()
    return pv.PolyData(smoothed.GetOutput())

## ======================== Parameters =================================
pixdim = [1,1,1]
vol_size = (128,128,128)
#vol_size = [192,192,192]
smoothing_factor = 0.3   #between 0 and 1
device = torch.device("cuda:0")

print("Checking database of patients ...")                                                                    
datalist = data.load_datalist(
    osp.join(dataRoot,datasetFilename),
    splits_=['test'],
    load_keys=['name', 'image'],
)

## ======================== Transforms =================================
keys = ['image']
monai.utils.misc.set_determinism(seed=218341029)

os.chdir(r"/home/your_dataset_path") #qui cambiare il nome della cartella del paziente corrente
#eventualmente fare ciclo for sui pazienti
trans = monai.transforms.Compose([
    monai.transforms.LoadImaged(keys),
    #clipLabeld(),
    monai.transforms.ToTensord(keys),
    monai.transforms.EnsureChannelFirstd(keys),
    monai.transforms.Orientationd(keys, axcodes='RAS'),
    monai.transforms.Spacingd(keys, pixdim=pixdim, mode=['bilinear']),
    monai.transforms.ScaleIntensityRanged(
        'image',
        a_min=-350,
        a_max=800,
        b_min=0,
        b_max=1,
    ),
    #monai.transforms.Flipd(keys, spatial_axis = 0),
    #monai.transforms.AsDiscreted(['label']),
    #monai.transforms.CropForegroundd(keys, source_key='image'),
    monai.transforms.EnsureTyped(keys),
    monai.transforms.SpatialPadd(keys, spatial_size=vol_size),
    monai.transforms.SaveImaged(
        keys,
        meta_keys=['image_meta_dict'],
        output_dir=dataRoot,
        output_postfix='trans',
        output_ext='.nii.gz',
        resample=False,
        output_dtype=np.float32,
        data_root_dir=dataRoot,
        separate_folder=False,
    )

])

print("Creating dataset ...")
ds = monai.data.Dataset(
    data=datalist,
    transform=trans
)
print("Creating data loader ...")
loader = monai.data.DataLoader(
        dataset = ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
)

# # Create Model, Loss, Optimizer, Trainer

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
print("Building neural network ...")
UNet_meatdata = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=monai.networks.layers.Norm.BATCH
)
model = monai.networks.nets.UNet(**UNet_meatdata).to(device)
print(f"Loading weights from path: {state_dict_path}")
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)
model.eval()

inferer = monai.inferers.SlidingWindowInferer(
    roi_size=vol_size,
    sw_batch_size=8,
    overlap=0.25,
    mode='constant',
    sw_device=device,
    device='cpu',
    progress=True,
)


for ii, data_ in enumerate(loader):
    print('ATTENZIONE: ', data_['name'][0])
    print(f"segmenting patient {ii}/{len(loader)} {data_['name']}")
    print('NIFTI PATH: ', os.path.join('/home/your_dataset_path',data_['name'][0]))
    #img_path = glob.glob
    #img_path = os.path.join('/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset',data_['image'][0],'/CT_nifti/',f'*_transf.nii.gz')
    #img_nii = nb.load(data_['image'])
    #affine = img_nii.affine
    #print('AFFINE: ', affine)
    image = data_['image'].to(device)
    with torch.no_grad():
        y_pred = inferer(image, model)

    y_pred = monai.data.decollate_batch(y_pred)
    assert len(y_pred) == 1
    y_pred = y_pred[0] # choose batch n. 0 from list
    # apply softmax and choose aorta channel (no background)
    y_pred = monai.transforms.Activations(softmax=True)(y_pred)[1]
    y_pred = monai.transforms.AsDiscrete(threshold=0.5)(y_pred).unsqueeze(0) # [1, H, W, D]
    name = os.path.split(data_['name'][0])[-1]
    print('patient name: ', name)

    ## ======================== Out directories =================================
    #dirOut_vtp = osp.join('/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset',data_['name'][0],SEGM_ROOT, logs_segm, out_name_vtp)
    dirOut_vtp = osp.join('/home/your_dataset_path',data_['name'][0], logs_segm, out_name_vtp)
    dirOut_nii = osp.join('/home/your_dataset_path',data_['name'][0], logs_segm, out_name_nii)
    dirOut_stl = osp.join('/home/your_dataset_path',data_['name'][0], logs_segm, out_name_stl)

    print(dirOut_vtp)
    if osp.exists(dirOut_vtp): 
        shutil.rmtree(dirOut_vtp)
        raise NameError('Output log dir already exists!')
    else:
        os.makedirs(dirOut_vtp)

    if osp.exists(dirOut_nii): 
        shutil.rmtree(dirOut_nii)
        raise NameError('Output log dir already exists!')
    else:
        os.makedirs(dirOut_nii)

    if osp.exists(dirOut_stl): 
        shutil.rmtree(dirOut_stl)
        raise NameError('Output log dir already exists!')
    else:
        os.makedirs(dirOut_stl)

    saver = monai.transforms.SaveImage(
        #output_dir='./',
        output_dir = dirOut_nii,
        output_postfix = 'seg',
        output_ext = '.nii.gz',
        resample=True,
        output_dtype=np.float32,
        separate_folder=False,
        #data_root_dir = dataRoot
    )

    data_ = monai.data.decollate_batch(data_)
    assert len(data_) == 1
    data_ = data_[0]
    saver(y_pred)

    y_pred_np = y_pred[0].detach().cpu().numpy()
    y_pred_np = monai.transforms.get_largest_connected_component_mask(y_pred_np)
    #y_pred_np = monai.transforms.utils.fill_holes(y_pred_np, connectivity=16)

    surface = marching_cubes(np.array(y_pred_np))
    surface.clear_data()
    #surface = surface.transform(data_['image_meta_dict']['affine'].numpy())
    surface = surface.transform(data_['image'].meta['affine'].numpy())
    surface = windowedSincSmooth(surface, iters=20, passband=10**(-4.*smoothing_factor))
    #surface.points[:,1] = -surface.points[:,1]
    #surface.points[:,0] = -surface.points[:,0]
    surface.points[:,:-1] = -surface.points[:,:-1] # LPS. Comment this to save in RAS (i think)

    #name = osp.basename(data_['image_meta_dict']['filename_or_obj']).split('.')[0]

    surface.save(osp.join(dirOut_vtp,f'{name}_seg.vtp'))
    surface.save(osp.join(dirOut_stl,f'{name}_seg.stl'))
    
