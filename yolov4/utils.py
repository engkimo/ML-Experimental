import os
import argparse
from easydict import EasyDict as edict


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='data/train.txt', help="train label path")
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())
    cfg.update(args)

    return edict(cfg)

def configration_init(num_cls, cfg):
    FIL=(num_cls+5)*3
    if cfg.is_tiny:
        os.system('rm cfg/yolov4-tiny.cfg')
        os.system('cp cfg/src-yolov4-tiny.cfg cfg/yolov4-tiny.cfg')
        os.system('mv cfg/yolov4-tiny.cfg cfg/yolov4-tiny.cfg.bak')
        os.system('cat cfg/yolov4-tiny.cfg.bak | sed -e "s/filters=255/filters={0}/g" -e "s/classes=80/classes={1}/g" > cfg/yolov4-tiny.cfg'.format(FIL, num_cls))
        os.system('rm cfg/*bak')
    else:
        os.system('rm cfg/yolov4.cfg')
        os.system('cp cfg/src-yolov4.cfg cfg/yolov4.cfg')
        os.system('mv cfg/yolov4.cfg cfg/yolov4.cfg.bak')
        os.system('cat cfg/yolov4.cfg.bak | sed -e "s/filters=255/filters={0}/g" -e "s/classes=80/classes={1}/g" > cfg/yolov4.cfg'.format(FIL, num_cls))
        os.system('rm cfg/*bak')

