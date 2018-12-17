import warnings
from argparse import ArgumentParser
import numpy as np
import cv2
import os
from os.path import join, dirname, exists, basename
from h5py import File as hfile
from scipy import misc
import ipdb
from tqdm import tqdm
from math import ceil, floor


def get_arguments():
    parser = ArgumentParser(
          description='Will convert 16bit grayscale images to colorject mapping as \
        in \"Multimodal Deep Learning for Robust RGB-D Object Recognition\"')
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument(
          "image_list", help="File containing the relative path to each file we want to convert")
    parser.add_argument("--colorjet", action="store_true")
    parser.add_argument("--h5", action="store_true")
    parser.add_argument("--ext", default="")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--buggy", action="store_true")
    parser.add_argument("--force_norm", action="store_true")
    parser.add_argument("--padding", action="store_true")
    parser.add_argument("--cropRatio", type=float,
                        help="Value between 0 and 1, represents the percentage of the image to keep")
    parser.add_argument("--useMask", action="store_true")
    args = parser.parse_args()
    return args


def smart_norm(img, force_norm, padding):
    # ipdb.set_trace()
    img = img.astype("float32")
    flat = img.ravel()
    ordered = np.argsort(flat)
    oldval = 0
    delta = 10
    for i, val in enumerate(ordered):
        diff = flat[val] - oldval
        if diff > delta:
            flat[ordered[i:]] -= (diff - delta)
        oldval = flat[val]
    max = img.max()
    if force_norm or max > 255.0:
        img = (255 / max) * img
    if padding:
        return add_padding(img)
    return img.astype('uint8')


def add_padding(img):
    imsz = img.shape
    mxdim = np.max(imsz)
    
    offs_col = (mxdim - imsz[1]) / 2
    offs_row = (mxdim - imsz[0]) / 2
    nchan = 1
    if (len(imsz) == 3):
        nchan = imsz[2]
    imgcanvas = np.zeros((mxdim, mxdim, nchan), dtype=img.dtype)
    imgcanvas[offs_row:offs_row + imsz[0], offs_col:offs_col +
                                                    imsz[1]] = img.reshape((imsz[0], imsz[1], nchan))
    # take rows
    if (offs_row):
        tr = img[0, :]
        br = img[-1, :]
        imgcanvas[0:offs_row, :, 0] = np.tile(tr, (offs_row, 1))
        imgcanvas[-offs_row - 1:, :, 0] = np.tile(br, (offs_row + 1, 1))
    # take cols
    if (offs_col):
        lc = img[:, 0]
        rc = img[:, -1]
        imgcanvas[:, 0:offs_col, 0] = np.tile(lc, (offs_col, 1)).transpose()
        imgcanvas[:, -offs_col - 1:,
        0] = np.tile(rc, (offs_col + 1, 1)).transpose()
    return imgcanvas


# Attention: the colorized depth image definition is: Red (close), blue(far)
# For TESTING pre-trained caffemodels use this function provided in this script
# Note that the opposite definition can be found in function depth2jet.cpp blue(close), red(far)
# When re-training the network it should not make a difference, but the second definition, is necessary
# for noise augmentation. Nan values (which we convert to 0 distance) are therefore dark blue and
# close objects are also blue.
def scaleit_experimental(img, invert, buggy, padding):
    img_mask = (img == 0)
    if img is None:
        return None
    istats = (np.min(img[img > 0]), np.max(img))
    if invert:
        imrange = 1.0 - (img.astype('float32') -
                         istats[0]) / (istats[1] - istats[0])
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                imrange = (img.astype('float32') -
                           istats[0]) / (istats[1] - istats[0])
            except Warning as e:
                print(
                      'DIVIDING BY ZERO!Python does not generate an exception but a simple warning.', e)
                # DIVIDING BY ZERO!Python does not generate an exception but a simple warning.
                # I decided to put the entire image to zero.
                imrange = (img.astype('float32') - istats[0])
    
    imrange[img_mask] = 0
    img = 255.0 * imrange
    imsz = imrange.shape
    mxdim = np.max(imsz)
    
    offs_col = (mxdim - imsz[1]) / 2
    offs_row = (mxdim - imsz[0]) / 2
    nchan = 1
    if (len(imsz) == 3):
        nchan = imsz[2]
    if padding:
        imgcanvas = np.zeros((mxdim, mxdim, nchan), dtype='uint8')
        imgcanvas[offs_row:offs_row + imsz[0], offs_col:offs_col +
                                                        imsz[1]] = img.reshape((imsz[0], imsz[1], nchan))
        # take rows
        if (offs_row):
            tr = img[0, :]
            br = img[-1, :]
            imgcanvas[0:offs_row, :, 0] = np.tile(tr, (offs_row, 1))
            imgcanvas[-offs_row - 1:, :, 0] = np.tile(br, (offs_row + 1, 1))
        # take cols
        if (offs_col):
            lc = img[:, 0]
            rc = img[:, -1]
            imgcanvas[:, 0:offs_col, 0] = np.tile(
                  lc, (offs_col, 1)).transpose()
            imgcanvas[:, -offs_col - 1:,
            0] = np.tile(rc, (offs_col + 1, 1)).transpose()
    else:
        imgcanvas = img.astype('uint8')
    # RESCALE
    return imgcanvas


IMSIZE = (256, 256)


def get_mask_crop(img, impath):
    # ipdb.set_trace()
    mask_path = join(dirname(impath), 'masks', basename(impath)[:-3] + "_mask.pbm")
    mask = cv2.imread(mask_path)
    if mask is None:
        return None
    c1 = mask.shape[0] / float(img.shape[0])
    c0 = mask.shape[1] / float(img.shape[1])
    ids = np.where(mask[:, :, 0] < 255)
    K = 30
    ly = [ids[1].min() / c0 - K, ids[1].max() / c0 + K]
    lx = [ids[0].min() / c1 - K, ids[1].max() / c1 + K]
    h = ly[1] - ly[0]
    w = lx[1] - lx[0]
    diff = abs(h - w)
    if h < w:
        ly[0] -= ceil(diff / 2.0)
        ly[1] += floor(diff / 2.0)
    elif w < h:
        lx[0] -= ceil(diff / 2.0)
        lx[1] += floor(diff / 2.0)
    return ly + lx


def get_center_crop(image, cropRatio):
    (w, h) = image.shape
    nw = 0.5 * w * cropRatio
    nh = max(0.5 * h * cropRatio, nw)
    nw = max(nw, nh)
    
    return (int(h / 2 - nh), int(h / 2 + nh), int(w / 2 - nw), int(w / 2 + nw))


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    
    output_dir = args.output_dir
    input_dir = args.input_dir
    with open(args.image_list) as tmp:
        images = tmp.readlines()
    emptyFiles = 0
    for i_path in tqdm(images):
        img_path = i_path.strip()
        full_path = join(input_dir, img_path)
        if args.h5:
            h = hfile(full_path)
            try:
                img = h['depth'][:]
            except:
                print("Couldn't load " + full_path)
                img = None
        else:
            img = cv2.imread(full_path, -1)
        if img is None:
            emptyFiles += 1
            print("Couldn't load %s" % full_path)
            continue
        # if args.get_single_channel:
        # img = img[:, :, 0]  # img = img[0,:,:]
        # newimg = smart_norm(img, args.force_norm)
        cArea = None
        if args.cropRatio:
            cArea = get_center_crop(img, args.cropRatio)
        if args.useMask:
            cArea = get_mask_crop(img, full_path)
        try:
            if cArea is not None:
                img = img[int(cArea[2]):int(cArea[3]), int(cArea[0]):int(cArea[1])]
            # new = smart_norm(img, args.force_norm, args.padding)
            new = scaleit_experimental(img, args.invert, args.buggy, args.padding)
            newimg = cv2.resize(new.astype('uint8'), IMSIZE, interpolation=cv2.INTER_CUBIC)
        except:
            print("Can't process " + full_path)
            emptyFiles += 1
            continue
        if args.colorjet:
            newimg = cv2.applyColorMap(newimg, cv2.COLORMAP_JET)
        outpath = join(output_dir, img_path + args.ext)
        outdir = dirname(outpath)
        if not exists(outdir):
            os.makedirs(outdir)
        cv2.imwrite(outpath, newimg)
    if emptyFiles > 0:
        print("Had to skip %d images" % emptyFiles)
