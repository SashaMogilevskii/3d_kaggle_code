"""
Скрипт который пересохраняет все маски в numpy format,
чтобы удобнее было работать с ними на сервере
"""


import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import ipyvolume as ipv
import nibabel as nib
from tqdm import tqdm

from preprocessing_img_for_segmentation import *

list_mask_nii = os.listdir(r"C:\KAGGLE\MEDICINE\data\segmentations")
os.mkdir("C:\KAGGLE\MEDICINE\data\segmentations_numpy")

for mask in tqdm(list_mask_nii):
    path_to_mask = os.path.join(r"C:\KAGGLE\MEDICINE\data\segmentations", mask)
    mask_data = create_3D_segmentations(path_to_mask)

    new_path_to_save = os.path.join(r"C:\KAGGLE\MEDICINE\data\segmentations_numpy", mask[:-4] + ".npy")
    np.save(new_path_to_save, mask_data)
