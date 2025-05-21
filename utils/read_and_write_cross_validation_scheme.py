import os
import pandas as pd

cross_validation_scheme_path = r"/.../your_LAA_db.csv"    
pat_folds = pd.read_csv(cross_validation_scheme_path, sep=",")
folds = 5

for fold in range(folds):
    # This will return a DataFrame containing only the rows where the column f"fold_{fold}" has the value "Train"
    train_elements = pat_folds.loc[pat_folds[fr"Fold_{fold}"] == 'Train'].dropna()["Id"]
    # This will return a DataFrame containing only the rows where the column f"fold_{fold}" has the value "Validation"
    val_elements = pat_folds.loc[pat_folds[fr"Fold_{fold}"] == 'Validation'].dropna()["Id"]
    
    os.chdir(r"/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset/")
    output_csv_file_tr = fr"train_elements_fold_{fold}.csv"
    output_csv_file_val = fr"val_elements_fold_{fold}.csv"
    output_csv_file_test = fr"test_elements.csv"

    # save as CSV file
    train_elements.to_csv(output_csv_file_tr, index=False)
    val_elements.to_csv(output_csv_file_val, index=False)

    print("File CSV salvato con successo.")

    del train_elements
    del val_elements
