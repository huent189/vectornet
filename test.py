import argparse
import os
from time import time

from utils.os import require_empty
from utils.logger import create_logger
from utils.enviroment import seed_all
from utils.checkpoint import load_model, save_model
from utils.image import random_patches

from data.gan import AlignedDataset
from data.supervised import PairedDataset

from model.zerodce import ZeroDCE
from model.gan import NoNormDiscriminator

from loss.zerodce import ZeroDCE_Loss
from loss.gan import GANLoss

from metrics.paired import psnr

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
    
def validate(loader, model, logger, metric_fn, writer, logs_dir):
    # Model to eval
    model.eval()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info('Validating')
    result = 0
    for batch_j, batch_data_val in enumerate(loader):
        x, y = batch_data_val
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred, _ = model.forward(x)
        result += metric_fn(y_pred, y)
        img_grid = torchvision.utils.make_grid([x[0], y_pred[0], y[0]])
        torchvision.utils.save_image(img_grid, os.path.join(logs_dir, f'img_{batch_j}.png'))
    logger.info(f'validation_loss: {result / len(loader)}')
    return result / len(loader)
def main(options):
    torch.autograd.set_detect_anomaly(False)
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
    val_data = PairedDataset(os.path.join( options.val_dir, "low"),
                                      os.path.join(options.val_dir, "high"))
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
    #***************#

    # logger.info('Total number of train samples: ~{}'.format(len(train_loader) * options.train_batch_size))
    logger.info('Total number of val samples: ~{}'.format(len(val_loader) * options.val_batch_size))

    #***************#
    generator = ZeroDCE().to(device)
    discriminator_A = NoNormDiscriminator(3, gpu_ids=options.gpu, n_layers=options.n_layers_D).to(device)
    discriminator_P = NoNormDiscriminator(3, gpu_ids=options.gpu, n_layers=options.n_layers_patchD).to(device)
    #***************#
    logger.info_trainable_params(generator)
    logger.info_trainable_params(discriminator_A)
    logger.info_trainable_params(discriminator_P)
    #***************#
    if options.init_model_filename:
        load_model(options.init_model_filename, generator, discriminator_A, discriminator_P)
    validate(val_loader, generator, logger, psnr, writer, logs_dir)
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
    options = parser.parse_args()

    return options

if __name__ == "__main__":
    opts = parse_args()
    seed_all(opts.seed)
    main(opts)