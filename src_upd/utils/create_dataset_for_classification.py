
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


def create_data_folds_for_classifier(data_path, label, debug=False):
    """base_logic"""
    # n_splits = 5
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, random_state=1771, shuffle=True)
    # data = pd.read_csv(os.path.join(data_path, "train.csv"))
    # data_meta = pd.read_csv(os.path.join(data_path, "train_series_meta.csv"))
    #
    # max_aortic_hu = data_meta.groupby('patient_id')['aortic_hu'].transform('max')
    # filtered_df = data_meta[data_meta['aortic_hu'] == max_aortic_hu]
    #
    # data = pd.merge(filtered_df, data, on='patient_id')

    """upd_logic"""
    # n_splits = 5
    # stratified_kfold = StratifiedKFold(n_splits=n_splits, random_state=1771, shuffle=True)
    # data = pd.read_csv(os.path.join(data_path, "train.csv"))
    # data_meta = pd.read_csv(os.path.join(data_path, "train_series_meta.csv"))
    # data_meta = data_meta[['patient_id', 'series_id']]


    """logic v_3"""

    n_splits = 5
    stratified_kfold = StratifiedKFold(n_splits=n_splits, random_state=1771, shuffle=True)
    data = pd.read_csv(os.path.join(data_path, "train.csv"))
    data_meta = pd.read_csv(os.path.join(data_path, "train_series_meta.csv"))
    data_meta = data_meta[['patient_id', 'series_id', 'aortic_hu']]
    data_size = pd.read_csv(os.path.join(data_path, "size_data.csv"))
    data = data.merge(data_meta, on='patient_id', how='inner').reset_index(drop=True)
    data = data.merge(data_size, on=['series_id','patient_id'],).reset_index(drop=True)
    df = data
    df_counts = df['patient_id'].value_counts().reset_index()
    df_counts.columns = ['patient_id', 'count']




    df_counts = df['patient_id'].value_counts().reset_index()
    df_counts.columns = ['patient_id', 'count']

    # Сначала обработаем записи, где patient_id встречается один раз
    single_occurrences = df_counts[df_counts['count'] == 1]['patient_id'].tolist()
    result = df[df['patient_id'].isin(single_occurrences)]

    # Теперь обработаем записи, где patient_id встречается два раза
    double_occurrences = df_counts[df_counts['count'] == 2]['patient_id'].tolist()

    for patient_id in double_occurrences:

        sub_df = df[df['patient_id'] == patient_id]
        count_layer_values = sub_df['count_layer'].values
        aortic_hu_values = sub_df['aortic_hu'].values
        series_id_values = sub_df['series_id'].values
        max_aortic_hu_index = aortic_hu_values.argmax()
        min_aortic_hu_index = aortic_hu_values.argmin()
        count_layer_for_max_aortic = count_layer_values[max_aortic_hu_index]
        count_layer_for_min_aortic = count_layer_values[min_aortic_hu_index]

        if max(aortic_hu_values) == min(aortic_hu_values) and max(count_layer_values) == min(count_layer_values):
            max_series_id = sub_df[sub_df['series_id'] == max(series_id_values)]
            result = pd.concat([result, max_series_id], ignore_index=True)

        elif max(aortic_hu_values) == min(aortic_hu_values):
            max_count_layer_row = sub_df[sub_df['count_layer'] == max(count_layer_values)]
            result = pd.concat([result, max_count_layer_row], ignore_index=True)

        elif max(count_layer_values) == min(count_layer_values):
            max_aortic_hu_row = sub_df[sub_df['aortic_hu'] == max(aortic_hu_values)]
            result = pd.concat([result, max_aortic_hu_row], ignore_index=True)
        elif max(count_layer_for_max_aortic, count_layer_for_min_aortic) <= 400:
            # Выбираем строку с максимальным count_layer
            max_count_layer_row = sub_df[sub_df['count_layer'] == max(count_layer_values)]
            result = pd.concat([result, max_count_layer_row], ignore_index=True)
        elif max(count_layer_for_max_aortic, count_layer_for_min_aortic) >= 1000:
            # Выбираем строку с максимальным aortic_hu
            max_aortic_hu_row = sub_df[sub_df['aortic_hu'] == max(aortic_hu_values)]
            result = pd.concat([result, max_aortic_hu_row], ignore_index=True)

        else:
            if count_layer_for_min_aortic - count_layer_for_max_aortic > 200:
                # Выбираем строку с меньшим aortic_hu
                min_aortic_hu_row = sub_df[sub_df['aortic_hu'] == min(aortic_hu_values)]
                result = pd.concat([result, min_aortic_hu_row], ignore_index=True)
            else:
                # Выбираем строку с большим aortic_hu
                max_aortic_hu_row = sub_df[sub_df['aortic_hu'] == max(aortic_hu_values)]
                result = pd.concat([result, max_aortic_hu_row], ignore_index=True)

    # Результат будет содержать только выбранные строки
    data = result

    if debug:
        try:
            df = pd.read_csv("../data_for_segmentation.csv")
        except:
            df = pd.read_csv("data_for_segmentation.csv")
        data = data[data.patient_id.isin(df.patient_id)]

    data = data.reset_index(drop=True)
    data['fold'] = -1

    # kidney
    if label == 'kidney':
        data['organ_type'] = data.apply(
            lambda row: 'healthy' if row['kidney_healthy'] == 1 else ('low' if row['kidney_low'] == 1 else 'high'),
            axis=1)



    # liver
    elif label == 'liver':
        data['organ_type'] = data.apply(
            lambda row: 'healthy' if row['liver_healthy'] == 1 else ('low' if row['liver_low'] == 1 else 'high'),
            axis=1)

    # spleen
    elif label == 'spleen':
        data['organ_type'] = data.apply(
            lambda row: 'healthy' if row['spleen_healthy'] == 1 else ('low' if row['spleen_low'] == 1 else 'high'),
            axis=1)

    # extravasation
    elif label == 'extravasation':
        data['organ_type'] = data.apply(
            lambda row: 'healthy' if row['extravasation_healthy'] == 1 else "injury",
            axis=1)

    # bowel
    elif label == 'bowel':
        data['organ_type'] = data.apply(
            lambda row: 'healthy' if row['bowel_healthy'] == 1 else "injury",
            axis=1)

    for fold_number, (train_index, test_index) in enumerate(stratified_kfold.split(data, data['organ_type'])):
        data.loc[test_index, 'fold'] = fold_number


    # update 10/9/2023
    # 3147 -> 4711
    """upd_logic"""
    # data = data.merge(data_meta, on='patient_id', how='inner').reset_index()
    return data


class ClassifierDataset(Dataset):
    def __init__(self,
                 data,
                 path_to_folder_with_img,
                 label,
                 channel,
                 is_train,
                 size):

        self.data = data
        self.path_to_folder_with_img = path_to_folder_with_img
        self.label = label
        self.channel = channel
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        path_to_load_crop = os.path.join(self.path_to_folder_with_img,
                                         self.label,
                                         str(row['patient_id']),
                                         str(row['series_id']), 'img.npy')
        if self.label == 'kidney':
            y_true = [row['kidney_healthy'], row['kidney_low'], row['kidney_high']]
        elif self.label == 'liver':
            y_true = [row['liver_healthy'], row['liver_low'], row['liver_high']]
        elif self.label == 'spleen':
            y_true = [row['spleen_healthy'], row['spleen_low'], row['spleen_high']]
        elif self.label == 'bowel':
            y_true = [row['bowel_healthy'], row['bowel_injury']]


        try:
            img = np.load(path_to_load_crop)

            if self.channel == 1:
                img = torch.Tensor(img).unsqueeze(0)

            elif self.channel == 2:
                path_to_load_crop_mask = os.path.join(self.path_to_folder_with_img,
                                                      self.label,
                                                      str(row['patient_id']),
                                                      str(row['series_id']), 'mask.npy')

                mask = np.load(path_to_load_crop_mask)
                img = np.stack([img, mask])
                img = torch.Tensor(img)
        except:
            if self.channel == 1:
                img = torch.zeros(self.size).unsqueeze(0)
            elif self.channel == 2:
                img = torch.zeros((2, self.size[0], self.size[1], self.size[2]))

            y_true = torch.zeros(len(y_true))


        return img, torch.Tensor(y_true)


if __name__ == '__main__':
    import yaml
    from box import Box

    with open("../my_configs/config_class.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = Box(config)

    data = create_data_folds_for_classifier(data_path=config.data_path,
                                            label=config.label,
                                            debug=False
                                            )

    train_data = data[data.fold != config.test_folds].reset_index(drop=True)

    new_dataset = ClassifierDataset(train_data,
                                    config.path_to_crop_img,
                                    config.label,
                                    is_train=True,
                                    channel=2,
                                    size=(16, 16, 16))

    sample = new_dataset[0]
