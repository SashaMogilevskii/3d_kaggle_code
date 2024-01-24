import yaml
import time
import os
import datetime
import warnings
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append("../pytorch3dunet/unet3d")
sys.path.append("../src/utils/")

import torch
import numpy as np


from torch import nn
from box import Box
from loguru import logger
from tqdm import tqdm
from torch.cuda import amp
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup


import losses
from base_scripts import set_seed, create_train_test_loader, create_fold_from_data
from my_losses import MyDiceLoss, CategoricalCrossEntropyLoss
from my_metrics import print_metric_MeanIoU

from model import ResidualUNet3D


date_now = datetime.datetime.now().strftime("%d_%B_%Y_%H_%M")

with open("configs/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    config = Box(config)

str_test_folds = [str(i) for i in config.test_folds]
path_save = os.path.join("experiment", date_now + '_' + '_'.join(str_test_folds) + '_' + config.model_name)

logger.add(f"{path_save}/info__{date_now}.log",
           format="<red>{time:YYYY-MM-DD HH:mm:ss}</red>| {message}")
logger.info(f"Folder with experiment - {path_save}")
file_name = __file__
logger.info(f'file for running: {file_name}')

with open(file_name, 'r') as file:
    code = file.read()
    logger.info(code)

config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(seed=config.seed)

logger.info("----------params----------")
for param in config:
    logger.info(f"{param}: {str(config[param])}")

if config.loss == "MyDiceLoss":
    loss = MyDiceLoss()
elif config.loss == "CategoricalCrossEntropyLoss":
    loss = CategoricalCrossEntropyLoss()
elif config.loss == 'CrossEntropy':
    loss = nn.CrossEntropyLoss()
elif config.loss == "DiceLose":
    loss = losses.DiceLoss(normalization='softmax')

criterion = loss


def train_loop(train_loader,
               valid_loader,
               epochs,
               model_name_for_save,
               scheduler,
               model,
               optimizer,
               scaler, loss):
    for epoch in range(1, epochs + 1):

        start = time.time()
        k = 0

        mloss_train, mloss_val = 0.0, 0.0

        list_y_true = None
        list_y_pred = None
        logger.info(f'Train model {model_name_for_save}')
        model.train()
        train_pbar = tqdm(train_loader, desc="Training", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for batch in train_pbar:

            X_batch = batch[0].to(config.device)
            masks_batch = batch[1].to(config.device)

            optimizer.zero_grad()
            with amp.autocast():
                pred_masks = model(X_batch)
                loss = criterion(pred_masks, masks_batch.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            mloss_train += loss.detach().item()

            cur_lr = f"LR : {optimizer.param_groups[0]['lr']:.2E}"
            if torch.cuda.is_available():
                train_pbar.set_postfix(gpu_load=f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB",
                                       loss=f"{loss.item():.4f}", lr=cur_lr)
            else:
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            del X_batch, masks_batch
            #####
            if config.debug:
                k += 1
                if k > 5: break
            ######

        del train_pbar

        # VALID
        model.eval()
        logger.info(f'Valid model')
        valid_pbar = tqdm(valid_loader, desc="Testing", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for batch in valid_pbar:
            X_batch = batch[0].to(config.device)
            masks_batch = batch[1].to(config.device)

            with torch.no_grad():
                pred_masks = model(X_batch)
                loss = criterion(pred_masks, masks_batch.long())
                mloss_val += loss.detach().item()
                if torch.cuda.is_available():
                    valid_pbar.set_postfix(gpu_load=f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB",
                                           loss=f"{loss.item():.4f}")
                else:
                    valid_pbar.set_postfix(loss=f"{loss.item():.4f}")

            if list_y_pred is None:
                list_y_pred = pred_masks.to('cpu').numpy()
                list_y_true = masks_batch.to('cpu').numpy()

            else:
                list_y_pred = np.vstack((list_y_pred, pred_masks.to('cpu').numpy()))
                list_y_true = np.vstack((list_y_true, masks_batch.to('cpu').numpy()))

            del X_batch, masks_batch

            ####
            if config.debug:
                k += 1
                if k > 8: break
            #####

        # Calculate metrics

        avg_train_loss = mloss_train / len(train_loader)
        avg_val_loss = mloss_val / len(valid_loader)

        logger.info(f'epoch: {epoch}')
        logger.info(cur_lr)
        logger.info("loss_train: %0.4f| loss_valid: %0.4f|" % (avg_train_loss, avg_val_loss))

        logger.info("METRICS")
        print_metric_MeanIoU(list_y_pred, list_y_true)

        elapsed_time = time.time() - start
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        logger.info(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'{path_save}/{model_name_for_save}_ep_{epoch}.pt')
        torch.save(model.state_dict(), f'{path_save}/{model_name_for_save}_last_epochs.pt')

        if config.debug and epoch > 10:
            break

        if config.device != 'cpu':
            torch.cuda.empty_cache()

        del valid_pbar


def main():
    data = create_fold_from_data(config.path_to_data)

    for test_fold in config.test_folds:
        model_name_for_save = config.model_name + '_fold_' + str(test_fold)
        logger.info(f'START FOLD {test_fold}')
        train_loader, valid_loader = create_train_test_loader(data=data,
                                                              test_fold=test_fold,
                                                              size_img=config.size_img,
                                                              path_to_folder_with_imgs=config.path_to_folder_with_imgs,
                                                              path_to_folder_with_masks=config.path_to_folder_with_masks,
                                                              typ_augm=config.type_augm,
                                                              batch_size=config.batch_size,
                                                              num_workers=config.num_workers
                                                              )
        if config.model_name == 'ResidualUNet3D':
            model = ResidualUNet3D(in_channels=1,
                                   out_channels=5,
                                   f_maps=config.f_maps,
                                   final_sigmoid=False)

        model.to(config.device)
        print(model.__doc__)
        all_steps = len(train_loader) * config.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=int(all_steps // 25),
                                                                       num_training_steps=all_steps)
        scaler = amp.GradScaler()
        logger.info(f"criterion - {config.loss}")
        logger.info(f"optimizer - {optimizer}")
        logger.info(f"sceduler - {scheduler}")

        train_loop(train_loader=train_loader,
                   valid_loader=valid_loader,
                   epochs=config.epochs,
                   model_name_for_save=model_name_for_save,
                   scheduler=scheduler,
                   model=model,
                   optimizer=optimizer,
                   scaler=scaler,
                   loss=criterion)

        logger.info(f'FINISH FOLD {test_fold}')
        logger.info(f'----------------------------')


main()
