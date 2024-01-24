# import os
# import pandas as pd
#
# from sklearn.model_selection import StratifiedKFold
# import torch.nn.functional as F
# import os
# import random
# from loguru import logger
# import time
# import numpy as np
# import pandas as pd
# import torch
# import random
# import volumentations as V
# from torch.utils.data import DataLoader, Dataset
# from scripts_read_imgs import process
#
#
# def create_data_folds_for_classifier(data_path, label, debug):
#     n_splits = 5
#     stratified_kfold = StratifiedKFold(n_splits=n_splits, random_state=1771, shuffle=True)
#     data = pd.read_csv(os.path.join(data_path, "train.csv"))
#     data_meta = pd.read_csv(os.path.join(data_path, "train_series_meta.csv"))
#
#     max_aortic_hu = data_meta.groupby('patient_id')['aortic_hu'].transform('max')
#     filtered_df = data_meta[data_meta['aortic_hu'] == max_aortic_hu]
#
#     data = pd.merge(filtered_df, data, on='patient_id')
#
#     if debug:
#         try:
#             df = pd.read_csv("../../src_upd/data_for_segmentation.csv")
#         except:
#             df = pd.read_csv("data_for_segmentation.csv")
#         data = data[data.patient_id.isin(df.patient_id)]
#
#     data = data.reset_index()
#     data['fold'] = -1
#
#     # kidney
#     if label == 'kidney':
#         data['organ_type'] = data.apply(
#             lambda row: 'healthy' if row['kidney_healthy'] == 1 else ('low' if row['kidney_low'] == 1 else 'high'),
#             axis=1)
#
#
#
#     # liver
#     elif label == 'liver':
#         data['organ_type'] = data.apply(
#             lambda row: 'healthy' if row['liver_healthy'] == 1 else ('low' if row['liver_low'] == 1 else 'high'),
#             axis=1)
#
#     # spleen
#     elif label == 'spleen':
#         data['organ_type'] = data.apply(
#             lambda row: 'healthy' if row['spleen_healthy'] == 1 else ('low' if row['spleen_low'] == 1 else 'high'),
#             axis=1)
#
#     # extravasation
#     elif label == 'extravasation':
#         data['organ_type'] = data.apply(
#             lambda row: 'healthy' if row['extravasation_healthy'] == 1 else "injury",
#             axis=1)
#
#     # bowel
#     elif label == 'bowel':
#         data['organ_type'] = data.apply(
#             lambda row: 'healthy' if row['bowel_healthy'] == 1 else "injury",
#             axis=1)
#
#     for fold_number, (train_index, test_index) in enumerate(stratified_kfold.split(data, data['organ_type'])):
#         data.loc[test_index, 'fold'] = fold_number
#
#     return data
#
#
# class ClassifierDataset(Dataset):
#     def __init__(self,
#                  data,
#                  seg_model,
#                  size,
#                  path_to_folder_with_imgs,
#                  label,
#                  padding,
#                  device,
#                  size_seg,
#                  ):
#         self.data = data
#         self.seg_model = seg_model
#         self.size = size
#         self.path_to_folder_with_imgs = path_to_folder_with_imgs
#         self.label = label
#         self.padding = padding
#         self.seg_aug = V.Compose([
#             V.Resize(size_seg, interpolation=3, resize_type=0, always_apply=True, p=1.0),
#         ], p=1.0)
#         self.aug = V.Compose([
#             V.Resize(self.size, interpolation=3, resize_type=0, always_apply=True, p=1.0),
#         ], p=1.0)
#         self.device = device
#
#         self.seg_model.eval()
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#
#         row = self.data.iloc[idx]
#         path_to_img = os.path.join(self.path_to_folder_with_imgs, str(row['patient_id']), str(row['series_id']))
#         img = process(data_path=path_to_img)
#         img_box = self.crop_box(img=img, id=row['patient_id'])
#
#         if self.label == 'kidney':
#             y_true = [row['kidney_healthy'], row['kidney_low'], row['kidney_high']]
#
#
#         elif self.label == 'liver':
#             y_true = [row['liver_healthy'], row['liver_low'], row['liver_high']]
#
#         elif self.label == 'spleen':
#             y_true = [row['spleen_healthy'], row['spleen_low'], row['spleen_high']]
#
#
#         elif self.label == 'bowel':
#             y_true = [row['bowel_healthy'], row['bowel_injury']]
#
#         if img_box is False:
#             logger.info(f'sample:patient_id, series_id {row["patient_id"], row["series_id"]} not find a label {self.label}')
#
#             empty_Tensor = torch.zeros(self.size).unsqueeze(0)
#             empty_y_true = torch.zeros(len(y_true))
#             return empty_Tensor, empty_y_true
#
#         data_for_class = {'image': img_box}
#         aug_data = self.aug(**data_for_class)
#
#         img_new = aug_data['image']
#
#         return torch.Tensor(img_new).unsqueeze(0), torch.Tensor(y_true)
#
#     def crop_box(self, img, id):
#
#         label_enc = {"kidney": 3,
#                      'bowel': 4,
#                      'spleen': 2,
#                      'liver': 1
#                      }
#
#         size_img = img.shape
#
#         data_for_aug = {'image': img}
#         aug_data = self.seg_aug(**data_for_aug)
#
#         img_new = aug_data['image']
#
#         self.seg_model.eval()
#         with torch.no_grad():
#             pred_mask = (self.seg_model)(torch.tensor(img_new).unsqueeze(0).unsqueeze(0).to(self.device))
#             pred_mask = pred_mask[0]
#
#         pred_mask = F.interpolate(
#             pred_mask.unsqueeze(0),
#             size=size_img,
#             mode='trilinear',
#             align_corners=False
#         )
#
#         pred_mask = pred_mask[0]
#         pred_mask = torch.argmax(pred_mask, dim=0)
#
#         class_to_extract = label_enc[self.label]
#
#         # if class_to_extract not in np.unique(pred_mask):
#         if class_to_extract not in torch.unique(pred_mask):
#             if self.device != 'cpu':
#                 torch.cuda.empty_cache()
#
#             del pred_mask
#             return False
#
#         indices = torch.where(pred_mask == class_to_extract)
#
#         min_d = max(indices[0].min() - self.padding, 0)
#         max_d = min(indices[0].max() + self.padding, pred_mask.shape[0] - 1)
#         min_h = max(indices[1].min() - self.padding, 0)
#         max_h = min(indices[1].max() + self.padding, pred_mask.shape[1] - 1)
#         min_w = max(indices[2].min() - self.padding, 0)
#         max_w = min(indices[2].max() + self.padding, pred_mask.shape[2] - 1)
#
#         img_box = img[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]
#
#         # если нужен баунд бокс маски
#         # bounding_box = pred_mask[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]
#         # bounding_box = np.where(bounding_box == class_to_extract, 1, 0)
#
#         if self.device != 'cpu':
#             torch.cuda.empty_cache()
#
#         del pred_mask, indices
#         return img_box
#
#
# if __name__ == '__main__':
#     import yaml
#     from box import Box
#
#     with open("../configs/config_class_щдввв.yaml", "r") as f:
#         config = yaml.load(f, Loader=yaml.SafeLoader)
#         config = Box(config)
#
#     data = create_data_folds_for_classifier(data_path=config.data_path,
#                                             label=config.label,
#                                             debug=config.debug
#                                             )
#
#     train_data = data[data.fold != config.test_fold].reset_index(drop=True)
