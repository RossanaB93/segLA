### Workflow
- Run 'createDatasetJson_with_cross_validation_scheme.py' (Transformed = False) to get the dataset_LA_cross_val_fold_k.json
- Run 'transformDataset.py' to apply trasnformations (cropping, resampling...) to the dataset (volume and segmentation)
- Run 'createDatasetJson_with_cross_validation_scheme.py' (Transformed = True) to get the dataset_LA_transf_cross_val_fold_k.json
- Run 'LASegmentation_with_cross_validation.py' to train the net with Kfold=0, Kfold=1, Kfold=2, Kfold=3, Kfold=4 changing output directory name
- Run 'test_LAsegmentation_with_cross_validation.py' to segment the test dataset with the trained net at each fold.

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

### LAsegmentation_with_cross_validation.py*
LAsegmentation_with_cross_validation.py trains a 3D U-net neural network on a given dataset. 

Output files:
•	.pth -> Best metric model saved as best_metric_model.pth
•	.png -> plot of loss function (training) and Dice coefficient (validation) as loss_and_dice.png
•	.txt files -> saved metrics (Dice coefficient, Hausdorff distance...)

### data_with_cross_validation_scheme.py
some helper function to manage data and other stuff

### test_LAsegmentation_with_cross_validation.py
test_LAsegmentation_with_cross_validation.py do segmentations of a dataset, based on the best_metric_model.pth (parameters of the U-net).

### notes
uses monai 1.2.0 installed using
```bash
conda create --name monai python=3.8
conda activate monai
pip install 'monai[all]'
```
