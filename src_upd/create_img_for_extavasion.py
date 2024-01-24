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
from preprocessing_img_for_segmentation import process


class Config:
    debug = False
    path_to_folder_with_data = r"C:\KAGGLE\\MEDICINE\\data\\"

    new_folder = 'img_for_extavasion'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size_seg = (192, 192, 192)


# Создаем дату, с изображениями по которым буду проходить и кропать части тела

data = pd.read_csv(os.path.join(Config.path_to_folder_with_data, "train.csv"))
data_meta = pd.read_csv(os.path.join(Config.path_to_folder_with_data, "train_series_meta.csv"))

# Дропаю лейблы, для которых было 2 варианта. Удаляю только вариант с наименьшим кол-вом данного вещества

data = pd.merge(data_meta, data, on='patient_id')
if Config.debug:
    # Оставляю только тех юзеров которые есть на локальном железе. Для дебага
    df = pd.read_csv("../src_upd/data_for_segmentation.csv")
    data = data[data.patient_id.isin(df.patient_id)]

data = data.reset_index(drop=True)
print('Shape', data.shape)

# Создаю репы в которых буду хранить новые изображения под лейблы
if not os.path.exists(os.path.join(Config.path_to_folder_with_data, Config.new_folder)):
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder))

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
    )[0][0].to('cpu').numpy()

    folder = os.path.join(Config.path_to_folder_with_data, Config.new_folder, patient_id, series_id)

    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, 'img_for_ext.npy'), img_tensor)

    gc.collect()
    torch.cuda.empty_cache()

    del img_tensor
    torch.cuda.empty_cache()
    logger.info(f"{i}, time:{time.time() - start}")
