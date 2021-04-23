import torch
import torch.utils.data as data
import cairosvg
from PIL import Image
import io
import numpy as np
import glob
import os
import cv2
class PathDataset(data.Dataset):
    def __init__(self, path, h, w):
        super(PathDataset, self).__init__()
        self.paths = sorted(glob.glob(path + "*"))
        self.w = w
        self.h = h
    def _preprocess_path(self, file_path, w, h):
        """
        return: x_0: image 3rd channel
                x_1: masked point
                y: giống x_0 ???
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            svg = f.read()

        r = 0
        s = [1, 1]
        t = [0, 0]
        svg = svg.format(w=w, h=h, r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity
        pid = 0
        num_paths = 0
        while pid != -1:
            pid = svg.find('path id', pid + 1)
            num_paths = num_paths + 1
        num_paths = num_paths - 1 # uncount last one

        path_id = np.random.randint(num_paths)
        svg_one = svg
        pid = len(svg_one)
        for c in range(num_paths):
            pid = svg_one.rfind('path id', 0, pid)
            if c != path_id:
                id_start = svg_one.rfind('>', 0, pid) + 1
                id_end = svg_one.find('/>', id_start) + 2
                svg_one = svg_one[:id_start] + svg_one[id_end:]

        # leave only one path
        y_png = cairosvg.svg2png(bytestring=svg_one.encode('utf-8'))
        y_img = Image.open(io.BytesIO(y_png))
        y = np.array(y_img)[:,:,3].astype(np.float) / max_intensity # [0,1]
        y[y > 0] = 1
        pixel_ids = np.nonzero(y)

        # select arbitrary marking pixel
        point_id = np.random.randint(len(pixel_ids[0]))
        px, py = pixel_ids[0][point_id], pixel_ids[1][point_id]

        y = np.reshape(y, [h, w, 1])
        x = np.zeros([h, w, 2])
        x[:,:,0] = s
        x[px,py,1] = 1.0
        return x, y
    def __getitem__(self, idx):
        p = self.paths[idx]
        x, y = self._preprocess_path(p, self.w, self.h)
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1).float()
        y = torch.from_numpy(y)
        y = y.permute(2, 0, 1).float()
        return x, y, p
    def __len__(self):
        return len(self.paths)

class OverlapDataset(data.Dataset):
    def __init__(self, path, h, w):
        super(OverlapDataset, self).__init__()
        self.paths = sorted(glob.glob(os.path.join(path, "*")))
        self.w = w
        self.h = h
    def preprocess_overlap(self, file_path, w, h):
        """
        return x: image
        y : intersection point
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            svg = f.read()
        
        r = 0
        s = [1, 1]
        t = [0, 0]

        svg = svg.format(w=w, h=h, r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity

        # while True:
        path_list = []        
        pid = 0
        num_paths = 0
        while pid != -1:
            pid = svg.find('path id', pid + 1)
            num_paths = num_paths + 1
        num_paths = num_paths - 1 # uncount last one
        
        for i in range(num_paths):
            svg_one = svg
            pid = len(svg_one)
            for j in range(num_paths):
                pid = svg_one.rfind('path id', 0, pid)
                if j != i:
                    id_start = svg_one.rfind('>', 0, pid) + 1
                    id_end = svg_one.find('/>', id_start) + 2
                    svg_one = svg_one[:id_start] + svg_one[id_end:]

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one.encode('utf-8'))
            y_img = Image.open(io.BytesIO(y_png))
            path = (np.array(y_img)[:,:,3] > 0)            
            path_list.append(path)

        y = np.zeros([h, w], dtype=np.int)
        for i in range(num_paths-1):
            for j in range(i+1, num_paths):
                intersect = np.logical_and(path_list[i], path_list[j])
                y = np.logical_or(intersect, y)

        x = np.expand_dims(s, axis=-1)
        y = np.expand_dims(y, axis=-1)

        return x, y
    def __getitem__(self, idx):
        p = self.paths[idx]
        x, y = self.preprocess_overlap(p, self.w, self.h)
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1).float()
        y = torch.from_numpy(y)
        y = y.permute(2, 0, 1).float()
        return x, y, p 
    def __len__(self):
        return len(self.paths)
class SingleDataset(data.Dataset):
    def __init__(self, path, h, w):
        super(SingleDataset, self).__init__()
        self.paths = sorted(glob.glob(path + "*"))
        self.w = w
        self.h = h
    def _preprocess_path(self, file_path, w, h):
        """
        return: x_0: image 3rd channel
                x_1: masked point
                y: giống x_0 ???
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            svg = f.read()

        r = 0
        s = [1, 1]
        t = [0, 0]
        svg = svg.format(w=w, h=h, r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity
        pid = 0
        num_paths = 0
        while pid != -1:
            pid = svg.find('path id', pid + 1)
            num_paths = num_paths + 1
        num_paths = num_paths - 1 # uncount last one

        path_id = np.random.randint(num_paths)
        svg_one = svg
        pid = len(svg_one)
        for c in range(num_paths):
            pid = svg_one.rfind('path id', 0, pid)
            if c != path_id:
                id_start = svg_one.rfind('>', 0, pid) + 1
                id_end = svg_one.find('/>', id_start) + 2
                svg_one = svg_one[:id_start] + svg_one[id_end:]

        # leave only one path
        y_png = cairosvg.svg2png(bytestring=svg_one.encode('utf-8'))
        y_img = Image.open(io.BytesIO(y_png))
        y = np.array(y_img)[:,:,3].astype(np.float) / max_intensity # [0,1]
        y[y > 0] = 1
        pixel_ids = np.nonzero(y)

        # select arbitrary marking pixel
        point_id = np.random.randint(len(pixel_ids[0]))
        px, py = pixel_ids[0][point_id], pixel_ids[1][point_id]

        y = np.reshape(y, [h, w, 1])
        x = np.zeros([h, w, 2])
        x[:,:,0] = s
        x[px,py,1] = 1.0
        return x, y
    def __getitem__(self, idx):
        p = self.paths[0]
        x, y = self._preprocess_path(p, self.w, self.h)
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1).float()
        y = torch.from_numpy(y)
        y = y.permute(2, 0, 1).float()
        return x, y, p + str(idx)
    def __len__(self):
        # return len(self.paths)
        return 128
if __name__ == "__main__":
    data = PathDataset("/content/kanji/val/", 64, 64)
    x, y, _ = data.__getitem__(0)
    pad = torch.zeros(1, 64, 64)
    path = torch.cat((x, pad), dim=0)
    path = path.permute(1,2,0)
    path = np.uint8(path.cpu() * 255)
    print(path.shape)
    y = y.squeeze(0)
    y = np.uint8(y * 255)
    cv2.imwrite('y.png', y)
    cv2.imwrite('x.png', path)