import os
import pandas as pd

cross_validation_scheme_path = r"/.../your_LAA_db.csv"    
#cross_validation_scheme_path = r"/mnt/Dati2/Ilaria Network/CODICE PYTHON/dataset/3D_UNet_LAA_db.csv"    
pat_folds = pd.read_csv(cross_validation_scheme_path, sep=",")
folds = 5
# Questo restituirà un DataFrame contenente solo le righe in cui la colonna fr"fold_{fold}"
# ha il valore "Train"
for fold in range(folds):
    train_elements = pat_folds.loc[pat_folds[fr"Fold_{fold}"] == 'Train'].dropna()["Id"]
    # Questo restituirà un DataFrame contenente solo le righe in cui la colonna "fold_0" 
    # ha il valore "Validation"
    val_elements = pat_folds.loc[pat_folds[fr"Fold_{fold}"] == 'Validation'].dropna()["Id"]
    # Questo restituirà un DataFrame contenente solo le righe in cui la colonna "fold_0" 
    # ha il valore "Test"
    test_elements = pat_folds.loc[pat_folds[fr"Fold_{fold}"] == 'Test'].dropna()["Id"]

    os.chdir(r"/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset/")
    output_csv_file_tr = fr"train_elements_fold_{fold}.csv"
    output_csv_file_val = fr"val_elements_fold_{fold}.csv"
    output_csv_file_test = fr"test_elements_fold_{fold}.csv"

    # Salva la serie come file CSV
    train_elements.to_csv(output_csv_file_tr, index=False)
    val_elements.to_csv(output_csv_file_val, index=False)
    test_elements.to_csv(output_csv_file_test, index=False)


    print("File CSV salvato con successo.")

    del train_elements
    del val_elements
    del test_elements
