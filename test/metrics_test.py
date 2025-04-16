import os 
import glob
import os.path as osp 
import nibabel as nb
import numpy as np
import pandas as pd

# definitions of the functions for the volumetric dice score, precision, and recall
def volumetric_dice_score(predicted, target, smooth=1):

    """
    Compute the Dice score for segmentation.

    Parameters:
    predicted (numpy.ndarray): Predicted segmentation mask.
    target (numpy.ndarray): Ground truth segmentation mask.

    Returns:
    Dice (float): Dice score.
    """

    # Ensure the input masks have the same shape
    if predicted.shape != target.shape:
        raise ValueError("Shape mismatch between predicted mask and ground truth mask.")

    intersection = np.sum(predicted * target)
    fake_union = np.sum(predicted) + np.sum(target)
    dice = (2. * intersection + smooth) / (fake_union + smooth)
   
    return dice

def precision_score(predicted, target):

    if predicted.shape != target.shape:
        raise ValueError("Shape mismatch between predicted mask and ground truth mask.")

    # True Positives (TP): Predicted pixels that are correctly classified as positive
    TP = np.sum(np.logical_and(predicted == 1, target == 1))
    # False Positives (FP): Predicted pixels that are incorrectly classified as positive
    FP = np.sum(np.logical_and(predicted == 1, target == 0))
    # Precision Score = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return precision

def recall_score(predicted, target, smooth=1):

    if predicted.shape != target.shape:
        raise ValueError("Shape mismatch between predicted mask and ground truth mask.")

    TP = np.sum(np.logical_and(predicted == 1, target == 1))
    # False Negatives (FN): Predicted pixels that are incorrectly classified as negative
    FN = np.sum(np.logical_and(predicted == 0, target == 1))
    # Recall Score = TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return recall

# path to the test data
dataRoot = r"/home/your_dataset_path/"
# set current dir
os.chdir(dataRoot) 
# choose the current fold
kfold = 1

# from the csv with the IDs of the patients of the test set for each fold to a dataframe
df = pd.read_csv(fr'/home/your_dataset_path/test_elements_fold_{kfold}.csv')
n_test_samples = len(df)

metrics_list =[]

for i in range(n_test_samples):
    
    prediction_path = osp.join(dataRoot,df.loc[i,'Id'],fr'test_K{kfold}_E500/nii_output_res') # nii_output_res: folder with the predicted segmentations
    target_path = osp.join(dataRoot,df.loc[i,'Id'],'CT_segmentation') # CT_segmentation: folder with the the ground truth
    
    p = nb.load(glob.glob(osp.join(prediction_path,'*_seg.nii.gz'))[0])
    t = nb.load(glob.glob(osp.join(target_path,'*_LA.nii.gz'))[0])
    
    predicted = p.get_fdata()
    target = t.get_fdata()
    
    dice = volumetric_dice_score(predicted, target)
    precision = precision_score(predicted,target)
    recall = recall_score(predicted,target)
    
    metrics_list.append((df.loc[i,'Id'],dice,precision,recall))

    print("Volumetric Dice score for "+df.loc[i,'Id'], dice.item())
    print("Precision score for "+df.loc[i,'Id'], precision.item())
    print("Recall score for "+df.loc[i,'Id'], recall.item())
    print('-'*40)

# from the lists with the scores to a single dataframe to build a table
df_metrics = pd.DataFrame(metrics_list,columns=['Patient_Id','Dice score', 'Precision','Recall'])

print(df)

# from the dataframe with the metrics to the csv
df_metrics.to_csv('metrics.csv',index=False)
# calculation of the mean values of each score
mean_values = (df_metrics['Dice score'].mean(),df_metrics['Precision'].mean(),df_metrics['Recall'].mean())

print('Mean value of Dice score for the fold '+str(kfold), str(mean_values[0]*100)+'%')
print('Mean value of Precision for the fold '+str(kfold), str(mean_values[1]*100)+'%')
print('Mean value of Recall for the fold '+str(kfold), str(mean_values[2]*100)+'%')



