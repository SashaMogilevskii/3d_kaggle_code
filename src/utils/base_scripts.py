# import os
# import random
#
# import numpy as np
# import pandas as pd
# import torch
# import random
# import volumentations as V
# from torch.utils.data import DataLoader, Dataset
# from scripts_read_imgs import (standardize_pixel_array,
#                                process,
#                                create_3D_segmentations)
#
#
# def set_seed(seed=1771):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True
#
#
# def create_fold_from_data(path_data, fold=5):
#     """
#     Проблема в том, что одному patient_id могут относиться 2 изображения.
#     Мне нужно разбить дату так, чтобы 2 изображения, которые относятся к одному patient_id
#     были в одном фолде.
#     :param path_data:
#     :param randome_state:
#     :param fold:
#     :return:
#     """
#     df = pd.read_csv(path_data)
#     df['count'] = df.groupby('patient_id')['patient_id'].transform('count')
#     df = df.sort_values(by='count', ascending=False)
#     df['FOLD'] = (df.index // (len(df) / fold)).astype(int)
#
#     return df
#
#
# def create_train_test_loader(data,
#                              test_fold,
#                              size_img,
#                              path_to_folder_with_imgs,
#                              path_to_folder_with_masks,
#                              typ_augm,
#                              batch_size,
#                              num_workers
#                              ):
#     train_data = data[data.FOLD != test_fold].reset_index(drop=True)
#     test_data = data[data.FOLD == test_fold].reset_index(drop=True)
#
#     train_dataset = SegmDataset(data=train_data,
#                                 is_train=True,
#                                 size=size_img,
#                                 path_to_folder_with_imgs=path_to_folder_with_imgs,
#                                 path_to_folder_with_masks=path_to_folder_with_masks,
#                                 typ_augm=typ_augm
#                                 )
#
#     test_dataset = SegmDataset(data=test_data,
#                                is_train=False,
#                                size=size_img,
#                                path_to_folder_with_imgs=path_to_folder_with_imgs,
#                                path_to_folder_with_masks=path_to_folder_with_masks,
#                                typ_augm=typ_augm
#                                )
#
#     train_loader = DataLoader(train_dataset,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               num_workers=num_workers,
#                               drop_last=True,
#                               )
#     valid_loader = DataLoader(test_dataset,
#                               batch_size=batch_size,
#                               shuffle=False,
#                               num_workers=num_workers,
#                               drop_last=False)
#
#     return train_loader, valid_loader
#
#
# class SegmDataset(Dataset):
#     def __init__(self,
#                  data,
#                  is_train,
#                  size,
#                  path_to_folder_with_imgs,
#                  path_to_folder_with_masks,
#                  typ_augm):
#
#         self.data = data
#         self.size = size
#         self.path_to_folder_with_imgs = path_to_folder_with_imgs
#         self.path_to_folder_with_masks = path_to_folder_with_masks
#
#         if typ_augm == 'v1':
#             if is_train:
#                 self.aug = V.Compose([
#                     V.Resize(self.size, interpolation=3, resize_type=0, always_apply=True, p=1.0),
#                 ], p=1.0)
#
#             else:
#                 self.aug = V.Compose([
#                     V.Resize(self.size, interpolation=3, resize_type=0, always_apply=True, p=1.0),
#                 ], p=1.0)
#
#         elif typ_augm == 'v2':
#
#             if is_train:
#                 self.aug = V.Compose([
#                     V.Resize(self.size, interpolation=3, resize_type=0, always_apply=True, p=1.0),
#                     V.Flip(0, p=0.5),
#                     V.Flip(1, p=0.5),
#                     V.Flip(2, p=0.5),
#                     V.GaussianNoise(var_limit=(0, 0.25), p=1)
#                     ], p=1.0)
#
#             else:
#                 self.aug = V.Compose([
#                     V.Resize(self.size, interpolation=3, resize_type=0, always_apply=True, p=1.0),
#                 ], p=1.0)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         path_to_img = os.path.join(self.path_to_folder_with_imgs, str(row['patient_id']), str(row['series_id']))
#         path_to_mask = os.path.join(self.path_to_folder_with_masks, f"{row['series_id']}.npy")
#
#         img = process(data_path=path_to_img)
#         mask = np.load(path_to_mask)
#
#         data = {'image': img, 'mask': mask}
#         aug_data = self.aug(**data)
#
#         img_new, mask_new = aug_data['image'], aug_data['mask']
#
#
#         return torch.Tensor(img_new).unsqueeze(0), torch.Tensor(mask_new)
#
# ## with load mask from nii format
# # class SegmDataset(Dataset):
# #     def __init__(self,
# #                  data,
# #                  is_train,
# #                  size,
# #                  path_to_folder_with_imgs,
# #                  path_to_folder_with_masks,
# #                  typ_augm):
# #
# #         self.data = data
# #         self.size = size
# #         self.path_to_folder_with_imgs = path_to_folder_with_imgs
# #         self.path_to_folder_with_masks = path_to_folder_with_masks
# #
# #         if typ_augm == 'v1':
# #             if is_train:
# #                 self.aug = V.Compose([
# #                     V.Resize(self.size, interpolation=3, resize_type=0, always_apply=True, p=1.0),
# #                 ], p=1.0)
# #
# #             else:
# #                 self.aug = V.Compose([
# #                     V.Resize(self.size, interpolation=3, resize_type=0, always_apply=True, p=1.0),
# #                 ], p=1.0)
# #
# #     def __len__(self):
# #         return len(self.data)
# #
# #     def __getitem__(self, idx):
# #         row = self.data.iloc[idx]
# #         path_to_img = os.path.join(self.path_to_folder_with_imgs, str(row['patient_id']), str(row['series_id']))
# #         path_to_mask = os.path.join(self.path_to_folder_with_masks, f"{row['series_id']}.nii")
# #
# #         img = process(data_path=path_to_img)
# #         mask = create_3D_segmentations(path_to_mask)
# #
# #         data = {'image': img, 'mask': mask}
# #         aug_data = self.aug(**data)
# #
# #         img_new, mask_new = aug_data['image'], aug_data['mask']
# #
# #         # add .unsqueeze(0)
# #         return torch.Tensor(img_new).unsqueeze(0), torch.Tensor(mask_new)
#
#
# # if __name__ == '__main__':
# #     df = create_fold_from_data(path_data="../data_for_segmentation.csv",
# #                                )
# #
# #     dataset = SegmDataset(data=df,
# #                           is_train=True,
# #                           size=(128, 128, 32),
# #                           typ_augm='v1',
# #                           path_to_folder_with_imgs=r"C:\KAGGLE\MEDICINE\data\train_images",
# #                           path_to_folder_with_masks=r"C:\KAGGLE\MEDICINE\data\segmentations")
# #
# #     img, mask = dataset[10]
# #     print(img.shape, mask.shape)
#
