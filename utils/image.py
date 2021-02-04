import torch
import random
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
