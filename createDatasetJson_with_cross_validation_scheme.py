import numpy as np
import random
import json
import csv  # Import the CSV module for working with CSV files
from pathlib import Path
import os 

dataset_path = r"/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset/"

transformed = False

baseDir = Path(dataset_path)
patient_dirs = list(baseDir.glob('*')) # crea una lista di oggetti path
patient_dirs = [dir_ for dir_ in patient_dirs if dir_.is_dir()]

print(patient_dirs)

os.chdir(baseDir)

folds = 5

for fold in range(folds):
    
    ordered_names_tr = []
    ordered_names_val = []
    ordered_names_test = []

    # Load CSV file for ordering
    with open(fr"train_elements_fold_{fold}.csv", newline='') as csvfile:  # Open CSV file
        reader = csv.reader(csvfile)  # Create a CSV reader object
        for row in reader:  # Iterate through rows in the CSV file
            ordered_names_tr.extend(row)  # Extend the list with names from each row

    ordered_names_tr = ordered_names_tr[1:] #così escludo i nomi delle colonne del csv file

    with open(fr"val_elements_fold_{fold}.csv", newline='') as csvfile:  # Open CSV file
        reader = csv.reader(csvfile)  # Create a CSV reader object
        for row in reader:  # Iterate through rows in the CSV file
            ordered_names_val.extend(row)  # Extend the list with names from each row

    ordered_names_val = ordered_names_val[1:] #così escludo i nomi delle colonne del csv file

    with open(fr"test_elements_fold_{fold}.csv", newline='') as csvfile:  # Open CSV file
        reader = csv.reader(csvfile)  # Create a CSV reader object
        for row in reader:  # Iterate through rows in the CSV file
            ordered_names_test.extend(row)  # Extend the list with names from each row

    ordered_names_test = ordered_names_test[1:] #così escludo i nomi delle colonne del csv file
    #print("Ordered Names:", ordered_names)

    os.chdir(r'/mnt/Dati2/Ilaria Network/CODICE PYTHON/')
    with open('datasetTemplate_LA.json') as json_file:
        data = json.load(json_file)

    temp = []

    os.chdir(dataset_path)

    for name in ordered_names_tr:
        print("Processing name:", name)
        dir_ = next((dir_ for dir_ in patient_dirs if dir_.stem == name), None)
        if dir_ is None:
            print("No directory found for name:", name)
        else:
            # Continua con il resto del codice

            img = list(dir_.glob("CT_nifti/*_trans.nii.gz")) if transformed else list(dir_.glob("CT_nifti/*[!_trans].nii.gz"))
            assert len(img) == 1, f"len(img): {len(img)}"
            img = img[0]

            seg = list(dir_.glob("CT_segmentation/*_trans.nii.gz")) if transformed else list(dir_.glob("CT_segmentation/*[!_trans].nii.gz"))
            assert len(seg) == 1, f"len(seg): {len(seg)}"
            seg = seg[0]

            temp.append({
                'name': name,
                'image': str(img.relative_to(baseDir)),
                'label': str(seg.relative_to(baseDir)),
            })
    data['numTraining'] = len(temp)
    data['training'] = temp

    temp = []
    for name in ordered_names_val:
        print("Processing name:", name)
        dir_ = next((dir_ for dir_ in patient_dirs if dir_.stem == name), None)
        if dir_ is None:
            print("No directory found for name:", name)
        else:
            # Continua con il resto del codice

            img = list(dir_.glob("CT_nifti/*_trans.nii.gz")) if transformed else list(dir_.glob("CT_nifti/*[!_trans].nii.gz"))
            assert len(img) == 1, f"len(img): {len(img)}"
            img = img[0]

            seg = list(dir_.glob("CT_segmentation/*_trans.nii.gz")) if transformed else list(dir_.glob("CT_segmentation/*[!_trans].nii.gz"))
            assert len(seg) == 1, f"len(seg): {len(seg)}"
            seg = seg[0]

            temp.append({
                'name': name,
                'image': str(img.relative_to(baseDir)),
                'label': str(seg.relative_to(baseDir)),
            })
    data['numValidation'] = len(temp)
    data['validation'] = temp

    temp = []
    for name in ordered_names_test:
        print("Processing name:", name)
        dir_ = next((dir_ for dir_ in patient_dirs if dir_.stem == name), None)
        if dir_ is None:
            print("No directory found for name:", name)
        else:
            # Continua con il resto del codice

            img = list(dir_.glob("CT_nifti/*_trans.nii.gz")) if transformed else list(dir_.glob("CT_nifti/*[!_trans].nii.gz"))
            assert len(img) == 1, f"len(img): {len(img)}"
            img = img[0]

            seg = list(dir_.glob("CT_segmentation/*_trans.nii.gz")) if transformed else list(dir_.glob("CT_segmentation/*[!_trans].nii.gz"))
            assert len(seg) == 1, f"len(seg): {len(seg)}"
            seg = seg[0]

            temp.append({
                'name': name,
                'image': str(img.relative_to(baseDir)),
                'label': str(seg.relative_to(baseDir)),
            })

    data['numTest'] = len(temp)
    data['test'] = temp

    json_string = json.dumps(data, indent=4)
    if not transformed:
        output_filename = baseDir / fr'dataset_LA_cross_val_fold_{fold}.json'
    else:
        output_filename = baseDir / fr'dataset_LA_transf_cross_val_fold_{fold}.json'
    os.chdir(r"/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset/")

    with open(output_filename, 'w') as outfile:
        outfile.write(json_string)

# Da riprendere per questo codice
# 3) controllare bene tutto e fare una copia di LAsegmentation per adattarla a una k-fold cross-validation controllata (e togliere il rumore gaussiano dalla data_augmentation)
# 4) capire cosa fa Transformed=True: è necessario?