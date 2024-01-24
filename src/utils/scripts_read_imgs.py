# import os
#
# import pydicom
# import cv2
# import numpy as np
# import nibabel as nib
#
#
# def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
#     """
#     Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
#     """
#     # Correct DICOM pixel_array if PixelRepresentation == 1.
#     pixel_array = dcm.pixel_array
#     if dcm.PixelRepresentation == 1:
#         bit_shift = dcm.BitsAllocated - dcm.BitsStored
#         dtype = pixel_array.dtype
#         pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
#     #         pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)
#
#     intercept = float(dcm.RescaleIntercept)
#     slope = float(dcm.RescaleSlope)
#     center = int(dcm.WindowCenter)
#     width = int(dcm.WindowWidth)
#     low = center - width / 2
#     high = center + width / 2
#
#     pixel_array = (pixel_array * slope) + intercept
#     pixel_array = np.clip(pixel_array, low, high)
#
#     return pixel_array
#
#
# def process(data_path="", size=512):
#     """
#     Read img and convert in np.array with standardization data.
#     :param data_path: path to folder with dcm files img
#     :param size: size to return img
#     :return:
#     """
#     lst_files = os.listdir(data_path)
#     lst_files = [int(x[:-4]) for x in lst_files]
#
#     if len(lst_files) > 1500:
#         step = 5
#     elif len(lst_files) > 1200:
#         step = 3
#     elif len(lst_files) > 800:
#         step = 2
#     else:
#         step = 1
#
#     imgs = []
#     for f in range(min(lst_files), max(lst_files) + 1):
#         path_to_files = os.path.join(data_path, f"{f}.dcm")
#
#         dicom = pydicom.dcmread(path_to_files)
#
#         img = standardize_pixel_array(dicom)
#         img = (img - img.min()) / (img.max() - img.min() + 1e-6)
#
#         if dicom.PhotometricInterpretation == "MONOCHROME1":
#             img = 1 - img
#
#         # if img.shape != (512, 512):
#         #     img = cv2.resize(img, (size, size))
#
#         imgs.append(img)
#
#     combined_array = np.stack(imgs, axis=0)
#
#     combined_array = np.transpose(combined_array, [1, 2, 0])
#
#     return combined_array
#
#
# def create_3D_segmentations(filepath, downsample_rate=1):
#     """
#     Стандартные лейблы были:
#     1 - liver
#     2 - spleen
#     3 - kidney_left
#     4 - kidney_right
#     5 - bowel
#     Изменю структуру, чтобы было более удобно на такой вариант
#
#     1 - liver
#     2 - spleen
#     3 - kidney (r & l)
#     4 - bowel
#     """
#     img = nib.load(filepath).get_fdata()
#     img = np.transpose(img, [1, 0, 2])
#     img = np.rot90(img, 1, (1, 2))
#     img = img[::-1, :, :]
#     img = np.transpose(img, [1, 0, 2])
#     img = img[::downsample_rate, ::downsample_rate, ::downsample_rate]
#     img = np.transpose(img, [1, 2, 0])
#     img = np.round(img).astype(int)
#
#     # 4 -> 3
#     img = np.where(img == 4, 3, img)
#     # 5 -> 4
#     img = np.where(img == 5, 4, img)
#
#     return img
