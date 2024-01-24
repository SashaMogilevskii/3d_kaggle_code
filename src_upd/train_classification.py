import sys
import gc
import yaml
import time
import os
import datetime
import warnings
import multiprocessing

sys.path.append("../pytorch3dunet/unet3d")
sys.path.append("../src_upd/utils/")

from utils.create_dataset_for_classification import create_data_folds_for_classifier, ClassifierDataset

from my_metrics import print_classification_metrics

from base_scripts import set_seed
from torch.utils.data import DataLoader, Dataset
from models3d.models.resnet import generate_model
from utils.my_losses import FocalLoss

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import numpy as np

from torch import nn
from box import Box
from loguru import logger
from tqdm import tqdm
from torch.cuda import amp
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup


def train_loop(train_loader,
               valid_loader,
               epochs,
               model_name_for_save,
               scheduler,
               model,
               optimizer,
               scaler, criterion):
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
            y_true = batch[1]

            # Проверяем, есть ли хотя бы один сэмпл, в котором все элементы равны 0
            # if torch.any(mask):
            if len([sample for sample in X_batch if not torch.all(sample == 0)]) != len(X_batch):
                pred_size = X_batch.shape
                X_batch = [sample for sample in X_batch if not torch.all(sample == 0)]
                if X_batch:
                    X_batch = torch.stack(X_batch).to(config.device)
                    y_true = torch.stack([sample for sample in y_true if not torch.all(sample == 0)])
                else:

                    scheduler.step()
                    del X_batch, y_true
                    gc.collect()

                    if config.device != 'cpu':
                        torch.cuda.empty_cache()
                    continue

                new_size = X_batch.shape
                print(pred_size, new_size)

            y_true = torch.argmax(y_true, dim=1).to(config.device)

            optimizer.zero_grad()

            with amp.autocast():
                pred_masks = model(X_batch)
                if config.loss == "CrossEntropy":
                    loss = criterion(pred_masks, y_true, weights=torch.tensor(config.weights))
                elif config.loss == "FocalLoss":
                    loss = criterion(F.softmax(pred_masks), y_true)
                elif config.loss == "BCEWithLogitsLoss":
                    assert ("Tyt ne doljen bit")
                    loss = criterion(pred_masks, y_true.unsqueeze(1).float())

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

            del X_batch, y_true
            gc.collect()

            if config.device != 'cpu':
                torch.cuda.empty_cache()
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
            y_true = batch[1].to(config.device)
            if len([sample for sample in X_batch if not torch.all(sample == 0)]) != len(X_batch):
                print("В батче есть вектора полностью из 0лей")
                pred_size = X_batch.shape
                X_batch = [sample for sample in X_batch if not torch.all(sample == 0)]
                if X_batch:
                    X_batch = torch.stack(X_batch).to(config.device)
                    y_true = torch.stack([sample for sample in y_true if not torch.all(sample == 0)])
                else:
                    continue
                new_size = X_batch.shape
                print(pred_size, new_size)

            with torch.no_grad():
                pred_masks = model(X_batch).to(config.device)

                if config.loss == "CrossEntropy":
                    loss = criterion(pred_masks, torch.argmax(y_true, dim=1), weights=torch.tensor(config.weights))
                elif config.loss == "FocalLoss":
                    loss = criterion(F.softmax(pred_masks), torch.argmax(y_true, dim=1))
                elif config.loss == "BCEWithLogitsLoss":
                    loss = criterion(pred_masks, torch.argmax(y_true, dim=1).unsqueeze(1).float())

                mloss_val += loss.detach().item()
                if torch.cuda.is_available():
                    valid_pbar.set_postfix(gpu_load=f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB",
                                           loss=f"{loss.item():.4f}")
                else:
                    valid_pbar.set_postfix(loss=f"{loss.item():.4f}")

            if list_y_pred is None:
                list_y_pred = F.softmax(pred_masks, dim=1).detach().to('cpu').numpy()

                list_y_true = y_true.to('cpu').numpy()

            else:
                list_y_pred = np.vstack((list_y_pred, F.softmax(pred_masks, dim=1).detach().to('cpu').numpy()))
                list_y_true = np.vstack((list_y_true, y_true.detach().to('cpu').numpy()))

            del X_batch, y_true
            gc.collect()
            if config.device != 'cpu':
                torch.cuda.empty_cache()

            ####
            if config.debug:
                k += 1
                if k > 8: break
            #####

        # Calculate metrics

        avg_train_loss = mloss_train / len(train_loader)
        avg_val_loss = mloss_val / len(valid_loader)

        logger.info(f'epoch: {epoch}')
        logger.info("loss_train: %0.4f| loss_valid: %0.4f|" % (avg_train_loss, avg_val_loss))

        logger.info("METRICS")
        print_classification_metrics(list_y_pred, list_y_true, config.label)

        elapsed_time = time.time() - start
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        logger.info(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'{path_save}/{model_name_for_save}_{config.label}_ep_{epoch}.pt')
        torch.save(model.state_dict(), f'{path_save}/{model_name_for_save}_{config.label}_last_epochs.pt')

        if config.debug and epoch > 10:
            break

        if config.device != 'cpu':
            torch.cuda.empty_cache()

        del valid_pbar



def main():
    data = create_data_folds_for_classifier(data_path=config.data_path,
                                            label=config.label,
                                            debug=config.debug
                                            )
    for test_fold in config.test_folds:
        model_name_for_save = config.model_name + str(config.model_depth) + '_fold_' + str(test_fold)

        logger.info(f'START FOLD {test_fold}')

        train_data = data[data.fold != test_fold].reset_index(drop=True)
        test_data = data[data.fold == test_fold].reset_index(drop=True)
        train_dataset = ClassifierDataset(train_data,
                                          config.path_to_crop_img,
                                          config.label,
                                          is_train=True,
                                          channel=config.channel,
                                          size = config.img_size)

        test_dataset = ClassifierDataset(test_data,
                                         config.path_to_crop_img,
                                         config.label,
                                         is_train=True,
                                         channel=config.channel,
                                         size=config.img_size)

        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  drop_last=True,
                                  )

        valid_loader = DataLoader(test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  drop_last=False)

        # if config.label in ["kidney", 'liver', 'spleen']:
        #     model = generate_model(config.model_depth, **{"n_input_channels": config.channel,
        #                                                   "n_classes": 3}).to(config.device)
        # else:
        #     model = generate_model(config.model_depth, **{"n_input_channels": config.channel,
        #                                                   "n_classes": 2}).to(config.device)
        from efficientnet_pytorch_3d import EfficientNet3D
        model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 3}, in_channels=1)
        model.to(config.device)

        print(model)
        all_steps = len(train_loader) * config.epochs

        if config.optm == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        elif config.optm == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                          weight_decay=config.weight_decay)

        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=all_steps,
                                      eta_min=0.00001)

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
                   criterion=criterion)

        logger.info(f'FINISH FOLD {test_fold}')
        logger.info(f'----------------------------')


if __name__ == '__main__':

    with open("my_configs/config_class.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config = Box(config)

    date_now = datetime.datetime.now().strftime("%d_%B_%Y_%H_%M")
    str_test_folds = [str(i) for i in config.test_folds]
    path_save = os.path.join("experiment",
                             date_now + '_' + config.label + '_' + '_'.join(
                                 str_test_folds) + '_' + config.model_name + str(
                                 config.model_depth))

    logger.add(f"{path_save}/info__{date_now}.log",
               format="<red>{time:YYYY-MM-DD HH:mm:ss}</red>| {message}")
    logger.info(f"Folder with experiment - {path_save}")
    file_name = __file__
    logger.info(f'file for running: {file_name}')

    with open(file_name, 'r', encoding='utf-8') as file:
        code = file.read()
        logger.info(code)

    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed=config.seed)

    logger.info("----------params----------")
    for param in config:
        logger.info(f"{param}: {str(config[param])}")

    if config.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif config.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif config.loss == "FocalLoss":
        criterion = FocalLoss(gamma=config.gamma,
                              weights=torch.Tensor(config.weights).to(config.device))

    main()
