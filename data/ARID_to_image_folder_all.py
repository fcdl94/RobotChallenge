import os
import argparse
from shutil import copyfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to transform ARID into Image Folder')
    parser.add_argument('--path', type=str, default="/home/fabio/robot_challenge/arid",
                        help="The input folder")

    args = parser.parse_args()
    path = args.path

    os.mkdir(path + "/all")

    for classname in os.listdir(path):
        if not "train" == classname and not "val" == classname and not "all" == classname:

            src_dir = path+"/"+classname

            for j, instdir in enumerate(sorted(os.listdir(src_dir))):
                dest = path+"/all/"+classname

                src = src_dir + "/" + instdir
                print("\t " + src)

                if not os.path.exists(dest):
                    os.mkdir(dest)

                for image in os.listdir(src):
                    if "crop" in image and "depth" not in image and "mask" not in image:
                        copyfile(src+"/"+image, dest+"/"+image)
