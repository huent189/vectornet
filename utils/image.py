import torch
import random
from PIL import Image
import numpy as np
import cv2
import os
def random_patches(tensor, num_patch, patchSize):
    patches = []
    w = tensor.size(3)
    h = tensor.size(2)
    for _ in range(num_patch):
        w_offset_1 = random.randint(0, max(0, w - patchSize - 1))
        h_offset_1 = random.randint(0, max(0, h - patchSize - 1))
        patches.append(tensor[:,:, h_offset_1:h_offset_1 + patchSize,
                    w_offset_1:w_offset_1 + patchSize])
    return patches
def to_nchw_numpy(image):
    if image.shape[3] in [1,2,3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image
def read_normed_numpy_img(path):
    im = Image.open(path)
    # im = np.array(im)[:,:,3].astype(np.float)
    # im = im / 255.0
    im = im / np.amax(im, axis=(0,1), keepdims=True)
    im = im.astype(np.float)
    return im
def read_normed_alpha_img(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im = im[:,:,-1]
    im = im / np.amax(im, axis=(0,1), keepdims=True)
    im = im.astype(np.float)
    return im 
def save_normed_im(imgs, path):
    if len(imgs.shape) == 4:
        txt, ext = os.path.splitext(path)
        for idx in range(0, imgs.shape[0], 8):
            if idx + 8 >= imgs.shape[0]:
              im = imgs[idx:]
            else:
              im = imgs[idx:idx+8]
            # print(im.shape)
            im = np.transpose(im, [0, 2, 3, 1])
            # im = im.reshape([im.shape[0],-1, im.shape[-1]])
            im = np.concatenate(im, axis=1)
            im = im * 255
            im = im.astype(np.uint8)
            new_path = f'{txt}_{idx}{ext}'
            cv2.imwrite(new_path, im)  
    elif len(imgs.shape) == 2 or imgs.shape[-1] == 3:
        imgs = imgs * 255
        imgs = imgs.astype(np.uint8)
        cv2.imwrite(path, imgs)
    else:
        raise NotImplementedError(str(imgs.shape))
def save_normed_rgb(imgs, p):
    assert len(imgs) == 2 or len(imgs) == 3, f'len error: {len(imgs)}'
    im = np.concatenate(imgs, axis=1)
    save_normed_im(im, p)
def read_normed_grayscale(fp, gamma=0, reverse=False):
    im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    im = cv2.resize(im, (64,64))
    if len(im.shape) > 2:
        if im.shape[-1] == 4:
            im = im[:,:,-1]
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #im must have black bg and text in white
    # if (np.mean(im) > 0.6):
    #     im = 255 - im
    if reverse:
        im = 255 - im 
    im = im / np.amax(im, axis=(0,1), keepdims=True)
    im = im ** (1 / gamma)
    im = im.astype(np.float)
    return im
if __name__ == '__main__':
    im = read_normed_numpy_img('0912d.png')
    im = im * 255
    im = im.astype(np.uint8)
    import cv2
    cv2.imwrite('test_read.png', im)