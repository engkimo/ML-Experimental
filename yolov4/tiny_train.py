import time
import logging
import os, sys, math
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter

from dataset import Yolo_dataset
from models import Yolov4
from utils import get_args, configration_init
from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator
from tiny_utils.cfg import Cfg
from losses.tiny_loss import Yolo_loss as Tiny_loss
from tiny_utils.darknet2pytorch import Darknet

def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, num_workers=0, save_cp=True, log_step=20, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True, collate_fn=val_collate)

    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained: :{config.pretrained}
    ''')

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Tiny_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes, is_tiny=config.is_tiny)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()
    for epoch in range(epochs):
        # model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_lr()[0] * config.batch
                                        })
                    logging.debug('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_lr()[0] * config.batch))

                pbar.update(images.shape[0])

            if cfg.use_darknet_cfg:
                eval_model = Darknet(cfg.cfgfile, inference=True)
            else:
                eval_model = Yolov4(cfg.pretrained, n_classes=cfg.classes, inference=True)
            # eval_model = Yolov4(yolov4conv137weight=None, n_classes=config.classes, inference=True)
            if torch.cuda.device_count() > 1:
                eval_model.load_state_dict(model.module.state_dict())
            else:
                eval_model.load_state_dict(model.state_dict())
            eval_model.to(device)
            evaluator = evaluate(eval_model, val_loader, config, device)
            del eval_model

            stats = evaluator.coco_eval['bbox'].stats
            writer.add_scalar('train/AP', stats[0], global_step)
            writer.add_scalar('train/AP50', stats[1], global_step)
            writer.add_scalar('train/AP75', stats[2], global_step)
            writer.add_scalar('train/AP_small', stats[3], global_step)
            writer.add_scalar('train/AP_medium', stats[4], global_step)
            writer.add_scalar('train/AP_large', stats[5], global_step)
            writer.add_scalar('train/AR1', stats[6], global_step)
            writer.add_scalar('train/AR10', stats[7], global_step)
            writer.add_scalar('train/AR100', stats[8], global_step)
            writer.add_scalar('train/AR_small', stats[9], global_step)
            writer.add_scalar('train/AR_medium', stats[10], global_step)
            writer.add_scalar('train/AR_large', stats[11], global_step)

            if save_cp:
                try:
                    # os.mkdir(config.checkpoints)
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f'Checkpoint {epoch + 1} saved !')
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f'failed to remove {model_to_remove}')

    writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[...,0] = boxes[...,0]*img_width
            boxes[...,1] = boxes[...,1]*img_height
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator




def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    
    print('log file path:' + log_file)
    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')



if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    configration_init(cfg.classes, cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if cfg.use_darknet_cfg:
        model = Darknet(cfg.cfgfile)
        model.load_weights(cfg.pretrained)
    else:
        model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device,
              num_workers=1)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

