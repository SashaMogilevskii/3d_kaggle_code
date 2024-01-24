import os

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

from preprocessing_img_for_segmentation import process
from torch.utils.data import DataLoader, Dataset
import volumentations as V


def create_fold_from_data(path_data, fold=5):
    """
    Проблема в том, что одному patient_id могут относиться 2 изображения.
    Мне нужно разбить дату так, чтобы 2 изображения, которые относятся к одному patient_id
    были в одном фолде.
    :param path_data:
    :param randome_state:
    :param fold:
    :return:
    """
    df = pd.read_csv(path_data)
    df['count'] = df.groupby('patient_id')['patient_id'].transform('count')
    df = df.sort_values(by='count', ascending=False)
    df['FOLD'] = (df.index // (len(df) / fold)).astype(int)

    return df


class SegmentationDataset(Dataset):
    def __init__(self,
                 data,
                 size,
                 is_train,
                 path_to_folder_with_imgs,
                 path_to_folder_with_masks,
                 typ_augm,
                 device):
        self.data = data
        self.size = size
        self.path_to_folder_with_imgs = path_to_folder_with_imgs
        self.path_to_folder_with_masks = path_to_folder_with_masks
        self.device = device
        self.is_train = is_train

        if typ_augm == 'v1':
            self.aug = V.Compose([
                V.Resize(self.size, interpolation=3, resize_type=0, always_apply=True, p=1.0),
            ], p=1.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path_to_img = os.path.join(self.path_to_folder_with_imgs, str(row['patient_id']), str(row['series_id']))
        path_to_mask = os.path.join(self.path_to_folder_with_masks, f"{row['series_id']}.npy")

        img = process(data_path=path_to_img)

        img_tensor = F.interpolate(
            torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
            size=self.size,
            mode='trilinear',
            align_corners=False
        )[0]

        mask = np.load(path_to_mask)
        upd_mask = self.aug(**{"mask": mask})['mask']

        return img_tensor, torch.Tensor(upd_mask)


def create_loaders_for_segmenatation(data,
                                     test_fold,
                                     size_img,
                                     path_to_folder_with_imgs,
                                     path_to_folder_with_masks,
                                     typ_augm,
                                     batch_size,
                                     num_workers,
                                     device):
    train_data = data[data.FOLD != test_fold].reset_index(drop=True)
    test_data = data[data.FOLD == test_fold].reset_index(drop=True)

    train_dataset = SegmentationDataset(data=train_data,
                                        size=size_img,
                                        path_to_folder_with_imgs=path_to_folder_with_imgs,
                                        path_to_folder_with_masks=path_to_folder_with_masks,
                                        typ_augm=typ_augm,
                                        is_train=True,
                                        device=device
                                        )
    test_dataset = SegmentationDataset(data=test_data,
                                      size=size_img,
                                      path_to_folder_with_imgs=path_to_folder_with_imgs,
                                      path_to_folder_with_masks=path_to_folder_with_masks,
                                      typ_augm=typ_augm,
                                      device=device,
                                      is_train=False
                                      )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True,
                              )
    valid_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=False)

    return train_loader, valid_loader


if __name__ == '__main__':
    df = create_fold_from_data(path_data="../data_for_segmentation.csv",
                               )

    dataset = SegmentationDataset(data=df,
                                  size=(192, 192, 192),
                                  typ_augm='v1',
                                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                  path_to_folder_with_imgs=r"C:\KAGGLE\MEDICINE\data\train_images",
                                  path_to_folder_with_masks=r"C:\KAGGLE\MEDICINE\data\segmentations_numpy")

    img, mask = dataset[10]
    print(img.shape, mask.shape)
