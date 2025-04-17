### Workflow
- Run 'read_and_write_cross_validation_scheme.py' to produce csv files containing the IDs of the patients to be included in the training, validation, and test sets at each fold
- Run 'createDatasetJson_with_cross_validation_scheme.py' (Transformed = False) to get the dataset_LA_cross_val_fold_k.json
- Run 'transformDataset.py' to apply trasnformations (cropping, resampling...) to the dataset (volume and segmentation)
- Run 'createDatasetJson_with_cross_validation_scheme.py' (Transformed = True) to get the dataset_LA_transf_cross_val_fold_k.json
- Run 'LASegmentation_with_cross_validation.py' to train the net with Kfold=0, Kfold=1, Kfold=2, Kfold=3, Kfold=4 changing output directory name
- Run 'test_LAsegmentation_with_cross_validation.py' to segment the test dataset with the trained net at each fold.
- Run 'metrics_test.py' to compute the metrics, such as the volumetric Dice score, the Precision, and the Recall of the test set for each fold.

### read_and_write_cross_validation_scheme.py
Input: a .csv file with a table containing "Training", "Validation", or "Test" for each patient depending on the current fold (for the first implementation of this framework, this .csv file is "UNet_all_phases_LAA_db.csv"). This script is for balancing the dataset, thus guaranteeing that at the end of the five folds the 10 phases of the CTs gated are equally processed by the UNet (there's no imbalance, such as 10 cases of 60% and 90 of 40%). The csv files is compiled upstreamly depending on the nummber and tha characteristics of the dataset.

Output: three csv files containing the IDs of the patient for the training, validation, and test of each fold.

### createDatasetJson_with_cross_validation_scheme.py
Create a .json file with all metadata about the dataset. In the .json file there will be:
•	Name -> name of the patient
•	Image -> path of the image (.gz)
•	Label -> path  of the segmentation mask (.nii)
Options in createDatasetJson.py:
•	transformed=False -> creates dataset.json of original dataset 
•	transformed=True -> creates dataset.json of transformed dataset (if present)
Output (transformed=False): dataset_LA_cross_val_fold_k.json
Output (transformed=True): dataset_LA_transf_cross_val_fold_k.json

The first run of the createDatasetJson_with_cross_validation_scheme.py is with the option transformed setted to False. 

### transformDataset.py
Create a transformed version of the dataset. Trasformations can include
- foreground cropping
- isotropic resampling to <pixdim>
- etc.
  
After the transforms run the createDatasetJson_with_cross_validation_scheme.py with the option ‘Transformed=True’ to create the .json file of the transformed dataset.

### segLA_cross_validation.py*
LAsegmentation_with_cross_validation.py trains a 3D U-net neural network on a given dataset. 

Output files:
- .pth -> Best metric model saved as best_metric_model.pth
- .png -> plot of loss function (training) and Dice coefficient (validation) as loss_and_dice.png
- .txt files -> saved metrics (Dice coefficient, Hausdorff distance...)

### test_LAsegmentation_with_cross_validation.py
test_LAsegmentation_with_cross_validation.py do segmentations of the test dataset for each fold, based on the best_metric_model.pth (parameters of the U-net).

### metrics_test.py
Compute the volumetric Dice score, Precision, and Recall for the test set of each fold. 
Output: a csv file containing a table with the before mentioned scores. Additionally, the script immediately print the values for each patient and also the mean values.

### data_with_cross_validation_scheme.py
some helper function to manage data and other stuff. 

### notes
uses monai 1.2.0 installed using
```bash
conda create --name monai python=3.8
conda activate monai
pip install 'monai[all]'
```
if an error is given, such as "monai not found", then change interpreter, and choose python 3.8.18 (always check the virtual environment!)

IMPORTANT! First finish the training with the 5-fold cross-validation then test each fold. Otherwise, the transformation performed with "transformDataset.py" must be run again!







