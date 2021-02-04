import argparse
import os
from time import time

from utils.os import require_empty
from utils.logger import create_logger
from utils.enviroment import seed_all
from utils.checkpoint import load_model, save_model

from data.vectornet import PathDataset, OverlapDataset

from model.vectornet import PathNet

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
def _make_gird(img, pred, trg):
    save_input = img[0,0]
    save_input[img[0,1] == 1.0] = 0
    save_input = save_input.unsqueeze(0)
    img_grid = torchvision.utils.make_grid([save_input, pred[0], trg[0]])
    return img_grid
def validate(loader, model, logger, log_i, metric_fn, writer):
    # Model to eval
    model.eval()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info('Validating')
    result = 0
    for batch_j, batch_data_val in enumerate(loader):
        x, y, _ = batch_data_val
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = model.forward(x) 
        result += metric_fn(y_pred, y)
    writer.add_scalar("validation_loss", result / len(loader), global_step=log_i)
    img_grid = _make_gird(x, y_pred, y)
    writer.add_image('validate_output', img_grid, global_step=log_i)
    return result / len(loader)
def main(options):
    torch.autograd.set_detect_anomaly(False)
    logs_dir = options.logs_dir
    require_empty(logs_dir, recreate=options.overwrite)
    logging_filename = os.path.join(logs_dir, "train.log")
    save_model_filename = os.path.join(logs_dir, "model")
    require_empty(save_model_filename, recreate=options.overwrite)
    tboard_dir = os.path.join(logs_dir, "tboard")
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
    writer = SummaryWriter(log_dir=tboard_dir)
    logger.info("Called with parameters: {}".format(options.__dict__))
    
    #***************#
    train_data = PathDataset(options.train_dir, options.im_w, options.im_h)
    val_data = PathDataset(options.val_dir, options.im_w, options.im_h)
    train_loader = DataLoader(train_data, batch_size=options.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=options.val_batch_size, shuffle=True, num_workers=4)
    #***************#

    logger.info('Total number of train samples: ~{}'.format(len(train_loader) * options.train_batch_size))
    logger.info('Total number of val samples: ~{}'.format(len(val_loader) * options.val_batch_size))

    #***************#
    if options.path:
        model = PathNet(options.repeat_num, options.conv_hidden_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, weight_decay=options.weight_decay, betas=[0.5, 0.999])
    cp_dict = {"model": model,
                'optimizer': optimizer}
    #***************#
    logger.info_trainable_params(model)
    if options.init_model_filename:
        load_model(cp_dict, optimizer)
    #***************#
    if options.loss == 'l1':
        criterion = nn.L1Loss(reduce='mean')
    else:
        criterion = nn.MSELoss(reduce='mean')
    #***************#
    best_val_loss = 1000
    for epoch_i in range(options.epochs):
        logger.info('Training batch {}'.format(epoch_i))
        epoch_loss = 0
        start_time = time()
        model.train()
        for j, data in enumerate(train_loader):
            model.zero_grad()
            img, trg, _ = data
            img = img.to(device)
            trg = trg.to(device)
            # global discriminator
            pred = model(img)
            loss = criterion(pred, trg)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        logger.info(
            'loss : {loss_G:.4f}'.format(loss_G=epoch_loss/len(train_loader)))
        writer.add_scalar("train_loss", epoch_loss / len(loader), global_step=epoch_i)
        logger.info('Time  {}'.format(time() - start_time))
        img_grid = _make_gird(img, pred, trg)
        writer.add_image('train_output', img_grid, global_step=epoch_i)
        val_loss = validate(val_loader, model, logger, epoch_i, criterion, writer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(os.path.join(save_model_filename, 'best.pth'), cp_dict)
        if (epoch_i % options.batches_before_save) == 0:
            save_model(os.path.join(save_model_filename, f'epoch_{epoch_i}.pth'), cp_dict)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', action='append', help='GPU to use, can use multiple [default: use CPU].')

    parser.add_argument('-e', '--epochs', type=int, default=1, help='how many epochs to train [default: 1].')
    parser.add_argument('-b', '--train-batch-size', type=int, default=128, dest='train_batch_size',
                        help='train batch size [default: 128].')
    parser.add_argument('-B', '--val-batch-size', type=int, default=128, dest='val_batch_size',
                        help='val batch size [default: 128].')
    
    parser.add_argument('--log-dir-prefix', dest='logs_dir', default='logs',
                        help='path to root of logging location [default: /logs].')
    parser.add_argument('-m', '--init-model-file', dest='init_model_filename',
                        help='Path to initializer model file [default: none].')

    parser.add_argument('--batches_before_save', type=int, default=1024, dest='batches_before_save',
                        help='how many batches to run before saving the model [default: 1024].')

    parser.add_argument('--train-dir', required=True, dest='train_dir',
                        help='name of train fake folder', type=str)
    parser.add_argument('--val-dir', required=True, dest='val_dir',
                        help='name of validation folder', type=str)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')
    parser.add_argument('-w', '--overwrite', action='store_true', default=False,
                        help='If set, overwrite existing logs [default: exit if output dir exists].')
    parser.add_argument('--seed', type=int, default=256)

    parser.add_argument('--path', action='store_true', help='train pathnet')
    parser.add_argument('--im_w', type=str, help='image_width', default=64)
    parser.add_argument('--im_h', type=str, help='image_heigh', default=64)
    parser.add_argument('--conv_hidden_num', type=int, default=64,
                     choices=[64, 128, 256])
    parser.add_argument('--repeat_num', type=int, default=20,
                     choices=[16, 20, 32])
    parser.add_argument('--loss', type=str, default='l1',
                     choices=['l1','l2'])
    options = parser.parse_args()

    return options

if __name__ == "__main__":
    opts = parse_args()
    seed_all(opts.seed)
    main(opts)