import sys
sys.path.insert(0, "..")
import os
import cv2
from shutil import copyfile
from copy import copy
import surface.create_processed_depthimages as srf
from multiprocessing import Pool
import data.ROD_to_image_folder as tif


BASE_FOLDER = "/home/fabioc/dataset/rod"
SRC_FOLDER = os.path.join(BASE_FOLDER, "full")

DEST_FOLDER = os.path.join(BASE_FOLDER, "proc")


def sub_sampling(src=SRC_FOLDER, dst=DEST_FOLDER):
    if not os.path.exists(dst):
        os.mkdir(dst)

    depth = []
    rgb = []
    mask = []

    for obj in os.listdir(src):
        src_obj_path = os.path.join(src,obj)
        dst_obj_path = os.path.join(dst,obj)
        if not os.path.exists(dst_obj_path):
            os.mkdir(dst_obj_path)

        for instance in os.listdir(src_obj_path):
            src_inst_path = os.path.join(src_obj_path, instance)
            dst_inst_path = os.path.join(dst_obj_path, instance)
            if not os.path.exists(dst_inst_path):
                os.mkdir(dst_inst_path)
            
            for file in sorted(os.listdir(src_inst_path)):
                if "crop" in file:
                    ff = file.split("_")
                    if int(ff[3]) % 5 == 1:
                        copyfile(os.path.join(src_inst_path, file), os.path.join(dst_inst_path, file))
                        if "_crop" in file:
                            rgb.append((dst_inst_path, file))
                        elif "depth" in file:
                            depth.append((dst_inst_path, file))
                        elif "mask" in file:
                            mask.append((dst_inst_path, file))
    return depth, rgb, mask


def apply_surface(depth, rgb, mask):
    pool1 = Pool()
    pool1.map(srf.process_depthimage, depth)
    pool2 = Pool()
    pool2.map(srf.process_colorimage, rgb)
    pool3 = Pool()
    pool3.map(srf.process_colorimage, mask)

def apply_cropping_masks(path):
    img_path = path

    ff = img_path[1].split("_")
    ff[4] = "maskcrop.png"

    mask_path = [img_path[0], "_".join(ff)]

    img_path = os.path.join(img_path[0], img_path[1])
    mask_path = os.path.join(mask_path[0], mask_path[1])

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    nimg = copy(img)
    nimg[mask==0] = (0,0,0)

    cv2.imwrite(img_path, nimg)

if __name__ == "__main__":
    MASK = False 
    # subsample
    print("Subsampling the data")
    # depth, rgb, mask = sub_sampling()
    print("Applying surface")
    # apply surface
    # apply_surface(depth, rgb, mask)
    # apply cropping masks (are we sure?)
    print("Applying the masking to images")
    if MASK:
        pool = Pool()
        pool.map(apply_cropping_masks, rgb)
        pool2 = Pool()
        pool2.map(apply_cropping_masks, depth)

    print("divinding in test and train")
    # divide in train and test
    tif.split_rod(BASE_FOLDER, 1, "proc")


