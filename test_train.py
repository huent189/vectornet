import argparse
import os
from time import time

from utils.os import require_empty
from utils.logger import create_logger
from utils.enviroment import seed_all
from utils.checkpoint import load_model, save_model

from data.vectornet import PathDataset, OverlapDataset, SingleDataset

from model.vectornet import PathNet

from metrics.raster import iou

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn

def _make_gird(img, pred, trg):
    if img.shape[1] == 2:
        save_input = img[0,0]
        save_input[img[0,1] == 1.0] = 0
        save_input = save_input.unsqueeze(0)
    else:
        save_input = img[0]
    img_grid = torchvision.utils.make_grid([save_input, pred[0], trg[-1]])
    return img_grid
def validate(loader, model, logger, metric_fns, log_path):
    # Model to eval
    metric_result = [0] * len(metric_fns)
    model.eval()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info('Validating')
    for batch_j, batch_data_val in enumerate(loader):
        x, y, p = batch_data_val
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = model(x)
            y_pred = torch.clamp(y_pred, 0, 1)
            print(p[0])
            for i in range(y_pred.shape[0]):
                img_grid = _make_gird(x[i], y_pred[i], x[i])
                # img_grid = y_pred[i]
                torchvision.utils.save_image(img_grid, os.path.join(log_path, os.path.split(p[i] + '.png')[-1]))
        for i in range(len(metric_fns)):
            metric = metric_fns[i][0]
            metric_result[i] += metric(y_pred, y)
    for i in range(len(metric_fns)):
        print(metric_fns[i][1], metric_result[i] / len(loader))
    
    
    return metric_result[-1] / len(loader)
def main(options):
    logs_dir = options.logs_dir
    require_empty(logs_dir, recreate=options.overwrite)
    logging_filename = os.path.join(logs_dir, "train.log")
    # train on multiple gpu
    if len(options.gpu) == 0:
        device = torch.device('cpu')
        prefetch_data = False
    elif len(options.gpu) == 1:
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(options.gpu)
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
        parallel = True
    logger = create_logger(logging_filename, options.verbose)
    logger.info("Called with parameters: {}".format(options.__dict__))
    
    #***************#
    if options.net == 'path':
        # val_data = PathDataset(options.val_dir, options.im_w, options.im_h)
        val_data = SingleDataset(options.val_dir, options.im_w, options.im_h)
    else:
        val_data = OverlapDataset(options.val_dir, options.im_w, options.im_h)
    val_loader = DataLoader(val_data, batch_size=options.val_batch_size, shuffle=True, num_workers=4)
    #***************#

    logger.info('Total number of val samples: ~{}'.format(len(val_loader) * options.val_batch_size))

    #***************#
    if options.net == 'path':
        model = PathNet(options.repeat_num, options.conv_hidden_num, input_channel=2).to(device)
    else:
        model = PathNet(options.repeat_num, options.conv_hidden_num, last_activation='sigmoid', input_channel=1).to(device)
    cp_dict = {"model": model
                # 'optimizer': optimizer
                }
    #***************#
    logger.info_trainable_params(model)
    if options.init_model_filename:
        load_model(options.init_model_filename, cp_dict)
    #***************#
    metrics = [[iou, "iou"]]
    val_loss = validate(val_loader, model, logger, metrics, logs_dir)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', action='append', help='GPU to use, can use multiple [default: use CPU].')

    parser.add_argument('-B', '--val-batch-size', type=int, default=128, dest='val_batch_size',
                        help='val batch size [default: 128].')
    
    parser.add_argument('--log-dir-prefix', dest='logs_dir', default='logs',
                        help='path to root of logging location [default: /logs].')
    parser.add_argument('-m', '--init-model-file', dest='init_model_filename',
                        help='Path to initializer model file [default: none].')

    parser.add_argument('--val-dir', required=True, dest='val_dir',
                        help='name of validation folder', type=str)

    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')
    parser.add_argument('-w', '--overwrite', action='store_true', default=False,
                        help='If set, overwrite existing logs [default: exit if output dir exists].')
    parser.add_argument('--seed', type=int, default=256)

    parser.add_argument('--net', type=str, help='type of net', choices=['path', 'overlap'])
    parser.add_argument('--im_w', type=str, help='image_width', default=64)
    parser.add_argument('--im_h', type=str, help='image_heigh', default=64)
    parser.add_argument('--conv_hidden_num', type=int, default=64,
                     choices=[64, 128, 256])
    parser.add_argument('--repeat_num', type=int, default=20,
                     choices=[16, 20, 32])
    options = parser.parse_args()

    return options

if __name__ == "__main__":
    opts = parse_args()
    seed_all(opts.seed)
    main(opts)