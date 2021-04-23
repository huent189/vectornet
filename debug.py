import numpy as np
import cv2
if __name__ == "__main__":
    img = cv2.imread('/content/kanji/jpg/074cf.jpg', 0)
    pts = img > (0.2 * 255)
    path_pixels = np.nonzero(pts)
    num_path_pixels = len(path_pixels[0]) 
    assert(num_path_pixels > 0)
    x_batch = np.zeros([img.shape[-2], img.shape[-1], 3])
    x_batch[:,:,0] = img
    for i in range(len(path_pixels[0])):
        px, py = path_pixels[0][i], path_pixels[1][i]
        x_batch[px,py,1] = 255
    cv2.imwrite('debug.png', x_batch)
    