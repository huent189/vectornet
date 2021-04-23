import argparse
import os
import time

from utils.os import require_empty
from utils.logger import create_logger
from utils.enviroment import seed_all
from utils.checkpoint import load_model, save_model
from utils.image import to_nchw_numpy, read_normed_numpy_img, save_normed_im, save_normed_rgb, read_normed_alpha_img

from data.vectornet import PathDataset, OverlapDataset

from model.vectornet import PathNet

from metrics.raster import iou

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import random
import cv2
import sklearn
import sklearn.neighbors
import skimage.measure
from datetime import datetime
import platform
from subprocess import call
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.misc
class Param(object):
    pass
def compute_accuracy(labels, pm):
    unique_labels = np.unique(labels)
    num_path_pixels = len(pm.path_pixels[0])

    acc_id_list = []
    acc_list = []
    for i in unique_labels:
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        i_label_map = np.zeros([pm.height, pm.width], dtype=np.bool)
        i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = True

        accuracy_list = []
        for j, stroke in enumerate(pm.path_list):
            intersect = np.sum(np.logical_and(i_label_map, stroke))
            union = np.sum(np.logical_or(i_label_map, stroke))
            accuracy = intersect / float(union)
            # print('compare with %d-th path, intersect: %d, union :%d, accuracy %.2f' % 
            #     (j, intersect, union, accuracy))
            accuracy_list.append(accuracy)

        id = np.argmax(accuracy_list)
        acc = np.amax(accuracy_list)
        # print('%d-th label, match to %d-th path, max: %.2f' % (i, id, acc))
        # consider only large label set
        # if acc > 0.1:
        acc_id_list.append(id)
        acc_list.append(acc)

    # print('avg: %.2f' % np.average(acc_list))
    return acc_list
def save_label_img(labels, unique_labels, num_labels, acc_avg, pm):
    print('number unique label:', unique_labels)
    sys_name = platform.system()

    file_path = os.path.basename(pm.file_path)
    file_name = os.path.splitext(file_path)[0]
    num_path_pixels = len(pm.path_pixels[0])
    
    cmap = plt.get_cmap('jet')    
    cnorm = colors.Normalize(vmin=0, vmax=num_labels-1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    
    label_map = np.ones([pm.height, pm.width, 3], dtype=np.float)
    label_map_t = np.ones([pm.height, pm.width, 3], dtype=np.float)
    first_svg = True
    target_svg_path = os.path.join(pm.model_dir, '%s_%d_%.2f.svg' % (file_name, num_labels, acc_avg))
    for color_id, i in enumerate(unique_labels):
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        color = np.asarray(cscalarmap.to_rgba(color_id))
        label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = color[:3]

        # save i label map
        i_label_map = np.zeros([pm.height, pm.width], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = pm.img[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]]
        _, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)
        i_label_map_path = os.path.join(pm.model_dir, 'tmp', 'i_%s_%d_%d.bmp' % (file_name, i, num_cc))
        save_normed_im(i_label_map, i_label_map_path)

        i_label_map = np.ones([pm.height, pm.width, 3], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = color[:3]
        label_map_t += i_label_map

        # vectorize using potrace
        color *= 255
        color_hex = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))
        
        if sys_name == 'Windows':
            potrace_path = os.path.join('potrace', 'potrace.exe')
            call([potrace_path, '-s', '-i', '-C'+color_hex, i_label_map_path])
        else:
            call(['potrace', '-s', '-i', '-C'+color_hex, i_label_map_path])
        
        i_label_map_svg = os.path.join(pm.model_dir, 'tmp', 'i_%s_%d_%d.svg' % (file_name, i, num_cc))
        if first_svg:
            copyfile(i_label_map_svg, target_svg_path)
            first_svg = False
        else:
            with open(target_svg_path, 'r') as f:
                target_svg = f.read()

            with open(i_label_map_svg, 'r') as f:
                source_svg = f.read()

            path_start = source_svg.find('<g')
            path_end = source_svg.find('</svg>')

            insert_pos = target_svg.find('</svg>')            
            target_svg = target_svg[:insert_pos] + source_svg[path_start:path_end] + target_svg[insert_pos:]

            with open(target_svg_path, 'w') as f:
                f.write(target_svg)

        # remove i label map
        # os.remove(i_label_map_path)
        # os.remove(i_label_map_svg)

    # set opacity 0.5 to see overlaps
    with open(target_svg_path, 'r') as f:
        target_svg = f.read()
    
    insert_pos = target_svg.find('<g')
    target_svg = target_svg[:insert_pos] + '<g fill-opacity="0.5">' + target_svg[insert_pos:]
    insert_pos = target_svg.find('</svg>')
    target_svg = target_svg[:insert_pos] + '</g>' + target_svg[insert_pos:]
    
    with open(target_svg_path, 'w') as f:
        f.write(target_svg)

    label_map_path = os.path.join(pm.model_dir, '%s_%.2f_%.2f_%d_%.2f.png' % (
        file_name, pm.sigma_neighbor, pm.sigma_predict, num_labels, acc_avg))
    save_normed_im(label_map, label_map_path)

    label_map_t /= np.amax(label_map_t)
    label_map_path = os.path.join(pm.model_dir, '%s_%.2f_%.2f_%d_%.2f_t.png' % (
        file_name, pm.sigma_neighbor, pm.sigma_predict, num_labels, acc_avg))
    save_normed_im(label_map_t, label_map_path)
def label(file_name, pm):
    start_time = time.time()
    working_path = os.getcwd()
    gco_path = os.path.join(working_path, 'gco/build')
    os.chdir(gco_path)

    pred_file_path = os.path.join(working_path, pm.model_dir, 'tmp', file_name + '.pred')
    print('ca', pred_file_path)        
    sys_name = platform.system()
    if sys_name == 'Windows':
        call(['Release/gco.exe', pred_file_path])
    else:
        call(['./gco', pred_file_path])
    os.chdir(working_path)

    # read graphcut result
    label_file_path = os.path.join(pm.model_dir, 'tmp', file_name + '.label')
    f = open(label_file_path, 'r')
    e_before = float(f.readline())
    e_after = float(f.readline())
    labels = np.fromstring(f.read(), dtype=np.int32, sep=' ')
    f.close()
    duration = time.time() - start_time
    print('%s: %s, labeling finished (%.3f sec)' % (datetime.now(), file_name, duration))

    return labels, e_before, e_after

def merge_small_component(labels, pm):
    knb = sklearn.neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knb.fit(np.array(pm.path_pixels).transpose())

    num_path_pixels = len(pm.path_pixels[0])

    for iter in range(2):
        # # debug
        # print('%d-th iter' % iter)
        
        unique_label = np.unique(labels)
        for i in unique_label:
            i_label_list = np.nonzero(labels == i)

            # handle duplicated pixels
            for j, i_label in enumerate(i_label_list[0]):
                if i_label >= num_path_pixels:
                    i_label_list[0][j] = pm.dup_rev_dict[i_label]

            # connected component analysis on 'i' label map
            i_label_map = np.zeros([pm.height, pm.width], dtype=np.float)
            i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = 1.0
            cc_map, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)

            # # debug
            # print('%d: # labels %d, # cc %d' % (i, num_i_label_pixels, num_cc))
            # plt.imshow(cc_map, cmap='spectral')
            # plt.show()

            # detect small pixel component
            for j in range(num_cc):
                j_cc_list = np.nonzero(cc_map == (j+1))
                num_j_cc = len(j_cc_list[0])

                # consider only less than 5 pixels component
                if num_j_cc > 4:
                    continue

                # assign dominant label of neighbors using knn
                for k in range(num_j_cc):
                    p1 = np.array([j_cc_list[0][k], j_cc_list[1][k]])
                    _, indices = knb.kneighbors([p1], n_neighbors=5)
                    max_label_nb = np.argmax(np.bincount(labels[indices][0]))
                    labels[indices[0][0]] = max_label_nb

                    # # debug
                    # print(' (%d,%d) %d -> %d' % (p1[0], p1[1], i, max_label_nb))

                    dup = pm.dup_dict.get(indices[0][0])
                    if dup is not None:
                        labels[dup] = max_label_nb

    return labels
def vectorize(pm):
    start_time = time.time()
    file_path = os.path.basename(pm.file_path)
    # file_name = os.path.splitext(file_path)[0]
    file_name = file_path

    # 1. label
    labels, e_before, e_after = label(file_name, pm)

    # 2. merge small components
    labels = merge_small_component(labels, pm)
    
    # # 2-2. assign one label per one connected component
    # labels = label_cc(labels, pm)

    # 3. compute accuracy
    # accuracy_list = compute_accuracy(labels, pm)

    unique_labels = np.unique(labels)
    num_labels = unique_labels.size        
    # acc_avg = np.average(accuracy_list)
    acc_avg = 0
    
    print('%s: %s, the number of labels %d' % (datetime.now(), file_name, num_labels))
    print('%s: %s, energy before optimization %.4f' % (datetime.now(), file_name, e_before))
    print('%s: %s, energy after optimization %.4f' % (datetime.now(), file_name, e_after))
    # print('%s: %s, accuracy computed, avg.: %.3f' % (datetime.now(), file_name, acc_avg))

    # 4. save image
    save_label_img(labels, unique_labels, num_labels, acc_avg, pm)
    duration = time.time() - start_time
    pm.duration_vect = duration
    
    # write result
    pm.duration += duration        
    print('%s: %s, done (%.3f sec)' % (datetime.now(), file_name, pm.duration))
    stat_file_path = os.path.join(pm.model_dir, file_name + '_stat.txt')
    with open(stat_file_path, 'w') as f:
        f.write('%s %d %.3f %.3f %.3f %.3f\n' % (
            file_path, num_labels,
            pm.duration_pred, pm.duration_map, 
            pm.duration_vect, pm.duration))
class Tester():
    def __init__(self,opts, device):
        self.path_net = PathNet(opts.repeat_num, opts.conv_hidden_num, input_channel=2).to(device)
        self.path_net.load_state_dict(torch.load(opts.path)['model'])
        self.path_net.eval()
        self.overlap_net = PathNet(opts.repeat_num, opts.conv_hidden_num, last_activation='sigmoid', input_channel=1).to(device)
        self.overlap_net.load_state_dict(torch.load(opts.overlap)['model'])
        self.overlap_net.eval()
        self.device = device
        self.out_dir = opts.out_dir
        self.sigma_neighbor = opts.sigma_neighbor
        self.neighbor_sample = opts.neighbor_sample
        self.sigma_predict = opts.sigma_predict
        self.b_num = opts.val_batch_size
        self.max_label = opts.max_label
        self.label_cost = opts.label_cost
        self.rng = np.random.RandomState()
    def extract_path(self, img):
        # pts = (img > 0.2)
        path_pixels = np.nonzero(img)
        num_path_pixels = len(path_pixels[0]) 
        assert(num_path_pixels > 0)

        y_batch = None
        x_all = None
        self.path_net.eval()
        for b in range(0,num_path_pixels,self.b_num):
            b_size = min(self.b_num, num_path_pixels - b)
            x_batch = np.zeros([b_size, img.shape[-2], img.shape[-1], 2])
            for i in range(b_size):
                x_batch[i,:,:,0] = img
                px, py = path_pixels[0][b+i], path_pixels[1][b+i]
                x_batch[i,px,py,1] = 1.0
        
            x_batch = torch.from_numpy(to_nchw_numpy(x_batch)).to(self.device).float()
            # print('s', x_batch.shape)
            with torch.no_grad():
                y_b = self.path_net(x_batch)
            y_b = torch.clamp(y_b, 0, 1)
            if y_batch is None:
                y_batch = y_b
                x_all = x_batch
            else:
                y_batch = torch.cat((y_b,y_batch), dim=0)
                x_all = torch.cat((x_batch, x_all), dim=0)
            torchvision.utils.save_image(torch.cat([x_batch[0], torch.zeros([1, 64, 64]).to(self.device)], dim=0), '/content/input.png')
            torchvision.utils.save_image(y_b[0], '/content/path.png')
        return y_batch.cpu().numpy(), path_pixels, x_all.cpu().numpy()
    def overlap(self, x):
        x = torch.from_numpy(x)
        x = x.unsqueeze(0).unsqueeze(0).float().to(self.device)
        self.overlap_net.eval()
        with torch.no_grad():
            y = self.overlap_net(x)
            y = torch.clamp(y, 0, 1)
            y[y < 0.7] = 0
        torchvision.utils.save_image(y[0], '/content/ov.png')
        return y.squeeze(0).squeeze(0).cpu().numpy(), x.squeeze(0).squeeze(0).cpu().numpy()
    def predict(self, img, file_name):
        pm = Param()
        start_time = time.time()
        paths, path_pixels, x = self.extract_path(img)
        num_path_pixels = len(path_pixels[0])
        duration = time.time() - start_time
        pm.duration_pred = duration
        pm.duration = duration
        ## save sample
        print('%s: %s, predict paths (#pixels:%d) through pathnet (%.3f sec)' % (datetime.now(), file_name, num_path_pixels, duration))
        pids = random.sample(range(num_path_pixels), 8)
        path_img_path = os.path.join(self.out_dir, '%s_1_path.png' % file_name)
        # save_normed_im(paths[pids], path_img_path)
        print('save path', paths.shape, x.shape)
        save_normed_rgb([paths, x], path_img_path) 
        ov, x = self.overlap(img)
        overlap_img_path = os.path.join(self.out_dir, '%s_2_overlap.png' % file_name)
        save_normed_rgb([x, x, ov], overlap_img_path)
        dup_id = num_path_pixels
        dup_dict = {}
        dup_rev_dict = {}
        for i in range(num_path_pixels):
            if ov[path_pixels[0][i], path_pixels[1][i]]:
                dup_dict[i] = dup_id
                dup_rev_dict[dup_id] = i
                dup_id += 1

        # write config file for graphcut
        start_time = time.time()
        tmp_dir = os.path.join(self.out_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        pred_file_path = os.path.join(tmp_dir, file_name+'.pred')
        f = open(pred_file_path, 'w')
        # info
        f.write(pred_file_path + '\n')
        f.write(self.out_dir + '\n')
        f.write('%d\n' % self.max_label)
        f.write('%d\n' % self.label_cost)
        f.write('%f\n' % self.sigma_neighbor)
        f.write('%f\n' % self.sigma_predict)
        # f.write('%d\n' % num_path_pixels)
        f.write('%d\n' % dup_id)

        radius = self.sigma_neighbor*2
        nb = sklearn.neighbors.NearestNeighbors(radius=radius)
        nb.fit(np.array(path_pixels).transpose())

        high_spatial = 100000
        for i in range(num_path_pixels-1):
            p1 = np.array([path_pixels[0][i], path_pixels[1][i]])
            pred_p1 = np.reshape(paths[i,:,:,:], [img.shape[-2], img.shape[-1]])

            # see close neighbors and some far neighbors (stochastic sampling)
            rng = nb.radius_neighbors([p1])
            num_close = len(rng[1][0])
            far = np.setdiff1d(range(i+1,num_path_pixels),rng[1][0])
            num_far = len(far)
            num_far = int(num_far * self.neighbor_sample)
            if num_far > 0:
                # print(far, num_far)
                far_ids = self.rng.choice(far, num_far)
                nb_ids = np.concatenate((rng[1][0],far_ids))
            else:
                nb_ids = rng[1][0]
            
            for rj, j in enumerate(nb_ids): # ids
                if j <= i:
                    continue                
                p2 = np.array([path_pixels[0][j], path_pixels[1][j]])
                if rj < num_close: d12 = rng[0][0][rj]
                else: d12 = np.linalg.norm(p1-p2, 2)            

            # for j in range(i+1, num_path_pixels): # see entire neighbors
            #     p2 = np.array([path_pixels[0][j], path_pixels[1][j]])
            #     d12 = np.linalg.norm(p1-p2, 2)
                
                pred_p2 = np.reshape(paths[j,:,:,:], [img.shape[-2], img.shape[-1]])
                pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5
                pred = np.exp(-0.5 * (1.0-pred)**2 / self.sigma_predict**2)

                spatial = np.exp(-0.5 * d12**2 / self.sigma_neighbor**2)
                f.write('%d %d %f %f\n' % (i, j, pred, spatial))

                dup_i = dup_dict.get(i)
                if dup_i is not None:
                    f.write('%d %d %f %f\n' % (j, dup_i, pred, spatial)) # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (i, dup_i, 0, high_spatial)) # shouldn't be labeled together
                dup_j = dup_dict.get(j)
                if dup_j is not None:
                    f.write('%d %d %f %f\n' % (i, dup_j, pred, spatial)) # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (j, dup_j, 0, high_spatial)) # shouldn't be labeled together

                if dup_i is not None and dup_j is not None:
                    f.write('%d %d %f %f\n' % (dup_i, dup_j, pred, spatial)) # dup_i < dup_j

        f.close()
        duration = time.time() - start_time
        print('%s: %s, prediction computed (%.3f sec)' % (datetime.now(), file_name, duration))
        pm.duration_map = duration
        pm.duration += duration
        
        pm.path_pixels = path_pixels
        pm.dup_dict = dup_dict
        pm.dup_rev_dict = dup_rev_dict
        pm.img = img
        pm.file_path = file_name
        pm.model_dir = self.out_dir
        pm.height = img.shape[-2]
        pm.width = img.shape[-1]
        pm.max_label = self.max_label
        pm.sigma_neighbor = self.sigma_neighbor
        pm.sigma_predict = self.sigma_predict

        return pm


    def test(self, img, file_name):
        param = self.predict(img, file_name)
        vectorize(param)

def main(options):
    if options.gpu is not None and len(options.gpu) == 1:
        device = 'cuda'
    else:
        device ='cpu'
    # device = torch.device('cuda:{}'.format(options.gpu[0]))
    device = torch.device(device)
    tester = Tester(options, device)
    im = read_normed_alpha_img(options.im)
    tester.test(im, os.path.split(options.im)[-1])
    # tester.overlap(im)
    # tester.extract_path(im)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', action='append', help='GPU to use, can use multiple [default: use CPU].')

    parser.add_argument('-B', '--val-batch-size', type=int, default=128, dest='val_batch_size',
                        help='val batch size [default: 128].')
    
    parser.add_argument('--log-dir-prefix', dest='out_dir', default='logs',
                        help='path to root of logging location [default: /logs].')

    parser.add_argument('--im', required=True, dest='im',
                        help='name of validation folder', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--overlap', type=str)
    parser.add_argument('--seed', type=int, default=256)
    parser.add_argument('--neighbor_sample', type=float, default=1)
    parser.add_argument('--sigma_neighbor', type=float, default=8.0)
    parser.add_argument('--sigma_predict', type=float, default=0.7)
    parser.add_argument('--max_label', type=int, default=26)
    parser.add_argument('--label_cost', type=int, default=0)
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
