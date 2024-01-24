"""
После обучение сегментатора идея о создание датасэта в котором модель на ходу будет кропать с изображения
 необходимую часть и подавать как getitem не сработала (Эпоха 2-3часа). Хочу сохранить каждый выкропленный
 мной орган отдельно

 Данный блокнот будет сохранять результат сегментации сразу ансамбля из k foldov. результат сохраню тоже,
 мб буду использовать в дальнейшем для обучение сегментаторов на псевдо-дате
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
import volumentations as V
from pytorch3dunet.unet3d.model import ResidualUNet3D
from preprocessing_img_for_segmentation import process
class Config:
    debug = True
    path_to_folder_with_data = r"C:\KAGGLE\\MEDICINE\\data\\"
    path_to_model_seg_fold_0 = "base_model_with_fold_1/02_October_2023_05_29_0_2_3_4_ResidualUNet3D/ResidualUNet3D_fold_0_last_epochs.pt"
    path_to_model_seg_fold_1 = "base_model_with_fold_1/01_October_2023_22_47_1_2_3_4_5_ResidualUNet3D/ResidualUNet3D_fold_1_last_epochs.pt"
    path_to_model_seg_fold_2 = "base_model_with_fold_1/02_October_2023_14_27_2_ResidualUNet3D/ResidualUNet3D_fold_2_ep_30.pt"
    path_to_model_seg_fold_3 = "base_model_with_fold_1/02_October_2023_14_28_3_ResidualUNet3D/ResidualUNet3D_fold_3_ep_30.pt"
    path_to_model_seg_fold_4 = "base_model_with_fold_1/02_October_2023_14_39_4_ResidualUNet3D/ResidualUNet3D_fold_4_ep_30.pt"


    new_folder = 'labels_img_after_pseudo'
    new_folder_for_segmentation = "pseudo_mask"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size_seg = (192, 192, 192)
    new_size = (128, 128, 128)
    padding = 9

# Создаем дату, с изображениями по которым буду проходить и кропать части тела

data = pd.read_csv(os.path.join(Config.path_to_folder_with_data, "train.csv"))
data_meta = pd.read_csv(os.path.join(Config.path_to_folder_with_data, "train_series_meta.csv"))
data = pd.merge(data, data_meta, on='patient_id')
# Дропаю лейблы, для которых было 2 варианта. Удаляю только вариант с наименьшим кол-вом данного вещества
# max_aortic_hu = data_meta.groupby('patient_id')['aortic_hu'].transform('max')
# filtered_df = data_meta[data_meta['aortic_hu'] == max_aortic_hu]
# data = pd.merge(filtered_df, data, on='patient_id')
if Config.debug:
    # Оставляю только тех юзеров которые есть на локальном железе. Для дебага
    df = pd.read_csv("../src_upd/data_for_segmentation.csv")
    data = data[data.patient_id.isin(df.patient_id)]

data = data.reset_index(drop=True)

# Создаю репы в которых буду хранить новые изображения под лейблы
if not os.path.exists(os.path.join(Config.path_to_folder_with_data, Config.new_folder)):
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder_for_segmentation))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder, "kidney"))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder, "liver"))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder, "spleen"))
    os.mkdir(os.path.join(Config.path_to_folder_with_data, Config.new_folder, "bowel"))

model_seg_fold_0 = ResidualUNet3D(in_channels=1,
                         out_channels=5,
                         f_maps=[32, 64, 128, 256, 512],
                         final_sigmoid=False).to(Config.device)
model_seg_fold_0.load_state_dict(torch.load(Config.path_to_model_seg_fold_0, map_location=Config.device))
model_seg_fold_0.to(Config.device)
model_seg_fold_0.eval()

model_seg_fold_1 = ResidualUNet3D(in_channels=1,
                         out_channels=5,
                         f_maps=[32, 64, 128, 256, 512],
                         final_sigmoid=False).to(Config.device)
model_seg_fold_1.load_state_dict(torch.load(Config.path_to_model_seg_fold_1, map_location=Config.device))
model_seg_fold_1.to(Config.device)
model_seg_fold_1.eval()

model_seg_fold_2 = ResidualUNet3D(in_channels=1,
                         out_channels=5,
                         f_maps=[32, 64, 128, 256, 512],
                         final_sigmoid=False).to(Config.device)
model_seg_fold_2.load_state_dict(torch.load(Config.path_to_model_seg_fold_2, map_location=Config.device))
model_seg_fold_2.to(Config.device)
model_seg_fold_2.eval()

model_seg_fold_3 = ResidualUNet3D(in_channels=1,
                         out_channels=5,
                         f_maps=[32, 64, 128, 256, 512],
                         final_sigmoid=False).to(Config.device)
model_seg_fold_3.load_state_dict(torch.load(Config.path_to_model_seg_fold_3, map_location=Config.device))
model_seg_fold_3.to(Config.device)
model_seg_fold_3.eval()

model_seg_fold_4 = ResidualUNet3D(in_channels=1,
                         out_channels=5,
                         f_maps=[32, 64, 128, 256, 512],
                         final_sigmoid=False).to(Config.device)
model_seg_fold_4.load_state_dict(torch.load(Config.path_to_model_seg_fold_4, map_location=Config.device))
model_seg_fold_4.to(Config.device)
model_seg_fold_4.eval()


def create_segmentation_mask(img):
    size_img = img.shape
    img_tensor = F.interpolate(
        torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda:0"),
        size=(192, 192, 192),
        mode='trilinear',
        align_corners=False
    )

    # fold_0
    pred_mask_0 = model_seg_fold_0(img_tensor)
    pred_mask_0 = F.interpolate(
        pred_mask_0.float(),
        size=size_img,
        mode='trilinear',
        align_corners=False
    )[0]
    pred_mask_0 = F.softmax(pred_mask_0, dim=0)
    pred_mask = pred_mask_0

    del pred_mask_0
    gc.collect()
    torch.cuda.empty_cache()

    # fold_1
    pred_mask_1 = model_seg_fold_1(img_tensor)
    pred_mask_1 = F.interpolate(
        pred_mask_1.float(),
        size=size_img,
        mode='trilinear',
        align_corners=False
    )[0]
    pred_mask_1 = F.softmax(pred_mask_1, dim=0)
    pred_mask += pred_mask_1
    del pred_mask_1
    gc.collect()
    torch.cuda.empty_cache()

    # fold_2
    pred_mask_2 = model_seg_fold_2(img_tensor)
    pred_mask_2 = F.interpolate(
        pred_mask_2.float(),
        size=size_img,
        mode='trilinear',
        align_corners=False
    )[0]
    pred_mask_2 = F.softmax(pred_mask_2, dim=0)
    pred_mask += pred_mask_2
    del pred_mask_2
    gc.collect()
    torch.cuda.empty_cache()

    # fold_3
    pred_mask_3 = model_seg_fold_3(img_tensor)
    pred_mask_3 = F.interpolate(
        pred_mask_3.float(),
        size=size_img,
        mode='trilinear',
        align_corners=False
    )[0]
    pred_mask_3 = F.softmax(pred_mask_3, dim=0)
    pred_mask += pred_mask_3
    gc.collect()
    torch.cuda.empty_cache()
    del pred_mask_3

    # fold_4
    pred_mask_4 = model_seg_fold_4(img_tensor)
    pred_mask_4 = F.interpolate(
        pred_mask_4.float(),
        size=size_img,
        mode='trilinear',
        align_corners=False
    )[0]
    pred_mask_4 = F.softmax(pred_mask_4, dim=0)
    pred_mask += pred_mask_4

    del pred_mask_4
    del img_tensor
    gc.collect()
    torch.cuda.empty_cache()

    pred_mask = pred_mask / 5.0
    pred_mask = torch.argmax(pred_mask.to(Config.device), dim=0)

    gc.collect()
    torch.cuda.empty_cache()

    return pred_mask

aug = V.Compose([
                V.Resize((192, 192, 192), interpolation=3, resize_type=0, always_apply=True, p=1.0),
            ], p=1.0)
for i, row in tqdm(data.iterrows()):
    start = time.time()

    patient_id = str(int(row["patient_id"]))
    series_id = str(int(row["series_id"]))
    logger.info(f'start {patient_id}/{series_id}')
    path_to_img = os.path.join(Config.path_to_folder_with_data, "train_images", patient_id, series_id)

    img = process(data_path=path_to_img)


    with torch.no_grad():

        average_mask = create_segmentation_mask(img)



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

                folder_for_pseudo_mask = os.path.join(Config.path_to_folder_with_data, Config.new_folder_for_segmentation, patient_id, series_id)
                os.makedirs(folder_for_pseudo_mask, exist_ok=True)
                np.save(os.path.join(folder, 'img.npy'), imb_box_for_classifier)
                np.save(os.path.join(folder, 'mask.npy'), mask_for_classifier)




            b = average_mask.to('cpu').numpy()
            upd_mask = aug(**{"mask": b})['mask']
            np.save(os.path.join(folder_for_pseudo_mask, 'mask_pseudo.npy'), upd_mask)

            gc.collect()
            torch.cuda.empty_cache()

    del average_mask
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"{i}, time:{time.time() - start}")
