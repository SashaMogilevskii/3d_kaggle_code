import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import torch.nn.functional as F

from preprocessing_img_for_segmentation import process


def create_data_folds_for_classifier(data_path, label, debug):
    n_splits = 5
    stratified_kfold = StratifiedKFold(n_splits=n_splits, random_state=1771, shuffle=True)
    data = pd.read_csv(os.path.join(data_path, "train.csv"))
    data_meta = pd.read_csv(os.path.join(data_path, "train_series_meta.csv"))

    max_aortic_hu = data_meta.groupby('patient_id')['aortic_hu'].transform('max')
    filtered_df = data_meta[data_meta['aortic_hu'] == max_aortic_hu]

    data = pd.merge(filtered_df, data, on='patient_id')

    if debug:
        try:
            df = pd.read_csv("../data_for_segmentation.csv")
        except:
            df = pd.read_csv("data_for_segmentation.csv")
        data = data[data.patient_id.isin(df.patient_id)]

    data = data.reset_index(drop=True)
    data['fold'] = -1


    if label == 'extravasation':
        data['organ_type'] = data.apply(
            lambda row: 'healthy' if row['extravasation_healthy'] == 1 else "injury",
            axis=1)
    else:
        ValueError('no correct label')


    for fold_number, (train_index, test_index) in enumerate(stratified_kfold.split(data, data['organ_type'])):
        data.loc[test_index, 'fold'] = fold_number

    return data


class ExtravasationDataset(Dataset):
    def __init__(self,
                 data,
                 path_to_folder_with_img,
                 size):
        self.data = data
        self.path_to_folder_with_img = path_to_folder_with_img
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        path_to_load_img = os.path.join(self.path_to_folder_with_img, 'img_for_extavasion',
                                         str(row['patient_id']),
                                         str(row['series_id']),
                                        "img_for_ext.npy'")
        y_true = [row['extravasation_healthy'], row['extravasation_injury']]

        img = np.load(path_to_load_img)
        return torch.Tensor(img).unsqueeze(0), torch.Tensor(y_true)


if __name__ == '__main__':
    import yaml
    from box import Box

    with open("../my_configs/config_extravasation.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = Box(config)

    data = create_data_folds_for_classifier(data_path=config.data_path,
                                            label=config.label,
                                            debug=config.debug
                                            )

    train_data = data[data.fold != config.test_fold].reset_index(drop=True)

    new_dataset = ExtravasationDataset(data=train_data,
                                       path_to_folder_with_img=config.path_to_crop_img,
                                       size=(96, 96, 96))

    sample = new_dataset[0]
