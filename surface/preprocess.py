import numpy as np
import cv2
import copy

IMSIZE = (224, 224)


def scale(image):
    s = image.shape
    if len(s) == 2:
        height, width = s
    else:
        height, width, channels = s
    diff = abs(width - height)
    if width > height:
        top = diff // 2
        bottom = top
        if diff % 2 == 1:
            top += 1
        
        return cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
    
    else:
        left = diff // 2
        right = left
        if diff % 2 == 1:
            right += 1
        
        return cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_REPLICATE)


def normalize(depthImg):
    mean = np.mean(depthImg)
    std = np.std(depthImg)
    out = depthImg.astype(np.float32) - mean
    min = -1.5
    max = 1.5
    out /= std
    out = np.clip(out, min, max)
    out = np.round(((out - min) / (max - min)) * 255).astype(np.uint8)
    return out


def resize(image, imsize=IMSIZE):
    image = cv2.resize(image, imsize, interpolation=cv2.INTER_CUBIC)
    return image


def apply_median_filter(depth_map):
    ksize = 5
    pad = int(ksize / 2)
    nonzero_threshold = 0
    
    tmp_depth = cv2.copyMakeBorder(depth_map, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    out = copy.copy(tmp_depth)
    
    y, x = np.nonzero(depth_map == 0)
    x.flags.writeable = True
    y.flags.writeable = True
    y += pad
    x += pad
    
    for idx in range(len(y)):
        i = y[idx]
        j = x[idx]
        nzero_vals = []
        for mi in range(i - pad, i + pad + 1):
            for mj in range(j - pad, j + pad + 1):
                val = tmp_depth[mi][mj]
                if val != 0:
                    nzero_vals.append(val)
        
        nzero_vals.sort()
        if len(nzero_vals) > nonzero_threshold:
            out[i][j] = np.median(nzero_vals)
    
    return cv2.medianBlur(out[pad:-pad, pad:-pad], 5)
