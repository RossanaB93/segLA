# ==========================================
# Imports
# ==========================================
import os
from pathlib import Path

import cv2
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

# ==========================================
# Constants
# ==========================================
LEFT_ATRIUM_LABEL = 1  # Label value for the left atrium in the segmentation

# ==========================================
# Utility functions
# ==========================================

def scan_dir(path: Path):
    """Returns the list of subdirectories in the given path."""
    return [f for f in path.iterdir() if f.is_dir()]

def calculate_ap_diameter(ct_nifti_path, seg_nifti_path, atrium_label):
    """
    Calculates the anterior-posterior (AP) diameter of the left atrium in a NIfTI image.

    Parameters:
        ct_nifti_path (str): Path to the CT image (NIfTI)
        seg_nifti_path (str): Path to the segmentation file (NIfTI)
        atrium_label (int): Label value of the left atrium

    Returns:
        tuple: (max_diameter_mm, slice_index)
    """
    # Load NIfTI files
    ct_img = nib.load(ct_nifti_path)
    ct_data = ct_img.get_fdata()

    seg_img = nib.load(seg_nifti_path)
    seg_data = seg_img.get_fdata()
    voxel_spacing = seg_img.header.get_zooms()

    max_area = 0
    best_slice_index = 0

    # Find slice with largest LA area
    for i in range(seg_data.shape[2]):
        binary_mask = (seg_data[:, :, i] == atrium_label).astype(np.uint8)
        area = binary_mask.sum()
        if area > max_area:
            max_area = area
            best_slice_index = i

    # Prepare binary mask for contour detection
    binary_slice = (seg_data[:, :, best_slice_index] == atrium_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, best_slice_index  # No segmentation found

    largest_contour = max(contours, key=cv2.contourArea)

    # Compute max Euclidean distance between any two points on the contour
    max_distance = 0
    max_points = ((0, 0), (0, 0))

    for i in range(len(largest_contour)):
        for j in range(i + 1, len(largest_contour)):
            dist = distance.euclidean(largest_contour[i][0], largest_contour[j][0])
            if dist > max_distance:
                max_distance = dist
                max_points = (largest_contour[i][0], largest_contour[j][0])

    # Convert to mm using pixel spacing
    diameter_mm = max_distance * voxel_spacing[0]
    return diameter_mm, best_slice_index

# ==========================================
# Main execution
# ==========================================

# Set dataset paths
segmentation_root = Path("/home/your_dataset_path/segmentations")
ct_root = Path("/home/your_dataset_path")

# Scan subject directories
subject_dirs = scan_dir(segmentation_root)

phase0_list = []
diameter_list = []

for subject_dir in subject_dirs:
    seg_folders = scan_dir(subject_dir)
    if not seg_folders:
        continue
    
    first_phase_files = os.listdir(seg_folders[0])
    if not first_phase_files:
        continue

    seg_file_path = seg_folders[0] / first_phase_files[0]
    ct_file_path = ct_root / subject_dir.name / "phase0.nii.gz"  # Adjust this if needed

    if not ct_file_path.exists():
        print(f"Missing CT for {ct_file_path}")
        continue

    try:
        diameter_mm, slice_index = calculate_ap_diameter(str(ct_file_path), str(seg_file_path), LEFT_ATRIUM_LABEL)
        phase0_list.append(first_phase_files[0])
        diameter_list.append(diameter_mm)
        print(f"{subject_dir.name}: AP diameter = {diameter_mm:.2f} mm (slice {slice_index})")
    except Exception as e:
        print(f"Error processing {subject_dir.name}: {e}")

# Save results
output_df = pd.DataFrame({
    "Phase0_File": phase0_list,
    "Diametro_AP_Atrio_Sinistro_mm": diameter_list
})
output_df.to_csv("diameters_ap.csv", index=False)
print("Saved diameters to diameters_ap.csv")
