"""
После обучение сегментатора идея о создание датасэта в котором модель на ходу будет кропать с изображения
 необходимую часть и подавать как getitem не сработала (Эпоха 2-3часа). Хочу сохранить каждый выкропленный
 мной орган отдельно
"""

import os
import gc
import sys
from loguru import logger
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional as F
sys.path.append("../pytorch3dunet/unet3d")
sys.path.append("../src_upd/utils/")
import torch
from pytorch3dunet.unet3d.model import ResidualUNet3D
from preprocessing_img_for_segmentation import process
class Config:
    debug = True
    path_to_folder_with_data = r"C:\KAGGLE\\MEDICINE\\data\\"
    path_to_model_1 = "ResidualUNet3D_fold_1_last_epochs.pt"
    new_folder = 'labels_img'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size_seg = (192, 192, 192)
    new_size = (128, 128, 128)
    padding = 4

# Создаем дату, с изображениями по которым буду проходить и кропать части тела

data = pd.read_csv(os.path.join(Config.path_to_folder_with_data, "train.csv"))
data_meta = pd.read_csv(os.path.join(Config.path_to_folder_with_data, "train_series_meta.csv"))

# Дропаю лейблы, для которых было 2 варианта. Удаляю только вариант с наименьшим кол-вом данного вещества
max_aortic_hu = data_meta.groupby('patient_id')['aortic_hu'].transform('max')
filtered_df = data_meta[data_meta['aortic_hu'] == max_aortic_hu]
data = pd.merge(filtered_df, data, on='patient_id')
if Config.debug:
    # Оставляю только тех юзеров которые есть на локальном железе. Для дебага
    df = pd.read_csv("../src_upd/data_for_segmentation.csv")
    data = data[data.patient_id.isin(df.patient_id)]

data = data.reset_index(drop=True)

# Создаю репы в которых буду хранить новые изображения под лейблы
if not os.path.exists(os.path.join(Config.path_to_folder_with_data, Config.new_folder)):
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder, "kidney"))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder, "liver"))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder, "spleen"))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder, "bowel"))

model_seg = ResidualUNet3D(in_channels=1,
                           out_channels=5,
                           f_maps=[32, 64, 128, 256, 512],
                           final_sigmoid=False).to(Config.device)
model_seg.load_state_dict(torch.load(Config.path_to_model_1, map_location=Config.device))
model_seg.to(Config.device)
model_seg.eval()

for i, row in tqdm(data.iterrows()):
    start = time.time()

    patient_id = str(int(row["patient_id"]))
    series_id = str(int(row["series_id"]))
    logger.info(f'start {patient_id}/{series_id}')
    path_to_img = os.path.join(Config.path_to_folder_with_data, "train_images", patient_id, series_id)

    img = process(data_path=path_to_img)
    size_img = img.shape

    img_tensor = F.interpolate(
        torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(Config.device),
        size=Config.size_seg,
        mode='trilinear',
        align_corners=False
    )

    with torch.no_grad():
        pred_mask = model_seg(img_tensor)

        del img_tensor
        gc.collect()
        torch.cuda.empty_cache()

        pred_mask = F.interpolate(
            pred_mask.float(),
            size=size_img,
            mode='trilinear',
            align_corners=False
        )[0]

        pred_mask = F.softmax(pred_mask, dim=0)
        pred_mask = torch.argmax(pred_mask, dim=0)
        average_mask = pred_mask

        mapping_label = [(1, "liver"),
                         (2, "spleen"),
                         (3, "kidney"),
                         (4, "bowel")
                         ]
        for ind_label, label in mapping_label:

            if ind_label not in torch.unique(average_mask):
                img_box = torch.zeros(Config.new_size)

            else:
                indices = torch.where(average_mask == ind_label)

                min_d = max(indices[0].min() - Config.padding, 0)
                max_d = min(indices[0].max() + Config.padding, average_mask.shape[0] - 1)
                min_h = max(indices[1].min() - Config.padding, 0)
                max_h = min(indices[1].max() + Config.padding, average_mask.shape[1] - 1)
                min_w = max(indices[2].min() - Config.padding, 0)
                max_w = min(indices[2].max() + Config.padding, average_mask.shape[2] - 1)
                img_box = img[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]
                bounding_box = average_mask[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]
                bounding_box = torch.where(bounding_box == ind_label, torch.tensor(1), torch.tensor(0))

                imb_box_for_classifier = F.interpolate(
                    torch.tensor(img_box, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(Config.device).float(),
                    size=Config.new_size,
                    mode='trilinear',
                    align_corners=False
                )[0][0].to('cpu').numpy()

                mask_for_classifier = F.interpolate(
                    torch.tensor(bounding_box, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(Config.device).float(),
                    size=Config.new_size,
                    mode='trilinear',
                    align_corners=False
                )[0][0].to('cpu').numpy()

                folder = os.path.join(Config.path_to_folder_with_data, Config.new_folder, label, patient_id, series_id)

                os.makedirs(folder, exist_ok=True)

                np.save(os.path.join(folder, 'img.npy'), imb_box_for_classifier)
                np.save(os.path.join(folder, 'mask.npy'), mask_for_classifier)

                gc.collect()
                torch.cuda.empty_cache()

    del pred_mask, average_mask
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"{i}, time:{time.time() - start}")
