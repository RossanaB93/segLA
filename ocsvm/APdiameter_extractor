import nibabel as nib
import numpy as np
import os 
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial import distance
from scipy.ndimage import label
import matplotlib.pyplot as plt
import pandas as pd

valore_atrio_sinistro = 1  # Valore che rappresenta l'atrio sinistro nel file

# All subdirectories in the current directory, not recursive.
def scan_dir(path):
    subfolders = [f for f in path.iterdir() if f.is_dir()]
    return subfolders

import nibabel as nib
import numpy as np
import cv2
from scipy.spatial import distance
from skimage.measure import label
import matplotlib.pyplot as plt

def calcola_diametro(ct_nifti, file_nifti, valore_atrio_sinistro):
    """
    Calcola il diametro antero-posteriore (AP) dell'atrio sinistro da un file NIfTI e visualizza le slice
    della superficie anteriore e posteriore.

    :param ct_nifti: percorso del file NIfTI contenente l'immagine CT
    :param file_nifti: percorso del file NIfTI contenente la segmentazione
    :param valore_atrio_sinistro: valore nella segmentazione corrispondente all'atrio sinistro
    :return: diametro antero-posteriore (AP) in millimetri
    """
    # Carica il file NIfTI del CT e della segmentazione
    img_ct = nib.load(ct_nifti)
    data_ct = img_ct.get_fdata()
    img = nib.load(file_nifti)
    header = img.header
    vox_dims = header.get_zooms()
    data = img.get_fdata()
    n_slices_z = data.shape[2]
    slice_maggiore_start = 0  # Inizializza il diametro

    # Trova la slice con il maggiore valore per l'atrio sinistro
    for i in range(n_slices_z):
        slice_data = data[:, :, i]
        
        if slice_data.sum() == 0:
            continue  # Salta le slice vuote

        # Crea una maschera binaria per l'atrio sinistro
        binary_slice = (slice_data == valore_atrio_sinistro).astype(int)

        # Trova le componenti connesse (labels)
        # num_components, labels = label(binary_slice)

        # if labels > 1:
        #     continue

        slice_maggiore = slice_data.sum()
        if slice_maggiore > slice_maggiore_start:
            slice_maggiore_start = slice_maggiore
            index_max = i

    # Binario per la slice con la massima area
    binary_slice_8bit = (data[:,:,index_max] > 0).astype(np.uint8) * 255

    # Trova i contorni
    contours, _ = cv2.findContours(binary_slice_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Trova la distanza massima tra i punti del contorno
    max_distance = 0
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            dist = distance.euclidean(contour[i][0], contour[j][0])
            if dist > max_distance:
                max_distance = dist 
                max_points = (contour[i][0], contour[j][0])

    # Overlay della segmentazione e del diametro
    ct_slice = data_ct[:, :, index_max]
    img_overlay = cv2.cvtColor(ct_slice.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_overlay, [contour], -1, (0, 255, 0), 2)
    cv2.line(img_overlay, tuple(max_points[0]), tuple(max_points[1]), (255, 0, 0), 2)

    # Visualizza immagine con overlay della segmentazione e diametro
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    plt.title("CT Slice with Segmentation and Diameter")
    plt.axis("off")
    plt.show()

    # Ritorna il diametro massimo in millimetri
    return max_distance * vox_dims[0], index_max


dirname = '/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset_all_phases/segmentations'
dirname_ct = '/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset_all_phases/'


p = Path(dirname)
folders_paths = [f for f in p.iterdir() if f.is_dir()]

phase0_list = []
diametro_list = []

for subfolder_path in folders_paths:
    segs = scan_dir(subfolder_path)
    all_phases = os.listdir(segs[0])
    phase0_list.append(all_phases[0])

    file_nifti = str(segs[0])+'/'+str(all_phases[0])
    diametro_list.append(calcola_diametro(file_nifti, valore_atrio_sinistro))
    print(f"Diametro antero-posteriore dell'atrio sinistro: {calcola_diametro(ct_nifti,file_nifti, valore_atrio_sinistro)} mm")

df = pd.DataFrame({"Diametro_AP_Atrio_Sinistro_mm": diametro_list})
df.to_csv("diameters_ap.csv")

ct_nifti = '/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset_all_phases/Paz_34/CT_nifti/fase_0 - 859117_Paz_34.nii.gz'
file_nifti = '/mnt/Dati2/Ilaria Network/CODICE PYTHON/balanced_dataset_all_phases/segmentations/Paz_34/nii_output_res/fase_0 - 859117_Paz_34_seg.nii.gz'
