import numpy as np
import time
import cv2
import os
import re
import operator
import preprocess
import copy
from multiprocessing import Pool
from argparse import ArgumentParser
import sys


def get_arguments():
    parser = ArgumentParser(description='Will convert 16bit grayscale images to Surface++ mapping')
    parser.add_argument("--dataset", type=str, default="sample")
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()
    return args


def process_colorimage(tu):
    dataset, f = tu
    # print( dataset + f)
    img = cv2.imread(dataset + str(f), cv2.IMREAD_UNCHANGED)
    surf = copy.copy(img)
    surf = preprocess.resize(surf)

    cv2.imwrite(dataset + f, surf)


def process_depthimage(tu):
    dataset, f = tu
    # print( dataset + f)
    img = cv2.imread(dataset + str(f), cv2.IMREAD_UNCHANGED)
    if img is None:
        print("!!!!!!!!!!!!!!!!!!!!ERROR \t" + dataset + f + " \t!!!!!!!!!!!!!")
        return

    out = copy.copy(img)
    n_missing = len(np.nonzero(out == 0)[0])
    while n_missing > 0:
        out = preprocess.apply_median_filter(out)
        n_missing = len(np.nonzero(out == 0)[0])
    img = out
    
    # row col channel
    rows, cols = img.shape
    # print(img.shape)
    
    surf = np.zeros((rows, cols, 3), dtype=np.float32)
    n = np.zeros(3, dtype=np.float32)
    
    img = np.array(img, dtype=np.float32)
    
    img = cv2.bilateralFilter(img, 25, 10, 10)
    
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    
    for x in range(0, rows - 1):
        for y in range(0, cols - 1):
            dzdx = (img[x + 1, y] - img[x - 1, y]) / 2.0
            dzdy = (img[x, y + 1] - img[x, y - 1]) / 2.0
            n[0] = -dzdx
            n[1] = -dzdy
            n[2] = 1.0
            n /= np.linalg.norm(n)
            surf[x, y, 0] = n.item(0)
            surf[x, y, 1] = n.item(1)
            surf[x, y, 2] = n.item(2)
    
    surf = surf[1:rows - 1, 1:cols - 1]
    
    gauss = cv2.GaussianBlur(surf, (0, 0), 3)
    cv2.addWeighted(surf, 1.5, gauss, -0.5, 0, surf)
    
    surf = preprocess.normalize(surf)
    
    # surf = preprocess.scale(surf)
    surf = preprocess.resize(surf)
    
    cv2.imwrite(dataset + f, surf)


args = get_arguments()

keywords_depth = {
    "ROD": "depth",
    "NYU": "png",
    "linemod": "depth",
    "sample": "png"
}

keywords_color = {
    "ROD": "_crop",
    "NYU": "jpg",
    "linemod": "color",
    "sample": "jpg"
}

BASE_ = "/home/fabioc/dataset/"
path = {
    "ROD": BASE_ + "rod/",
    "NYU": BASE_ + "nyu/",
    "linemod": BASE_ + "linemod/",
    "sample": BASE_ + "sample/"
}

# Dataset
dataset = path[args.dataset]
    
keyword_depth = keywords_depth[args.dataset]
keyword_color = keywords_color[args.dataset]

depth = []
color = []

print("Starting to pre-process images")
sys.stdout.flush()

for prefix in ["train/", "val/"]:
    
    classes = os.listdir(dataset + prefix)
    for c in classes:
        base = dataset + prefix + c + "/"
        files = os.listdir(base)
        for f in files:
            if keyword_depth in f:
                depth.append((base, f))
            elif keyword_color in f:
                color.append((base, f))

DEPTH = args.depth
if DEPTH:
    print("Starting with depth images....")
    pool = Pool()
    pool.map(process_depthimage, depth)
    print("..." + str(len(depth)) + " depth images processed")
else:
    print("Starting with color images....")
    pool = Pool()
    pool.map(process_colorimage, color)
    print("..." + str(len(color)) + " color images processed")


