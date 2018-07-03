import os
import argparse
from shutil import copyfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to transform ROD into Image Folder')
    parser.add_argument('--path', type=str, default="/home/lab2atpolito/FabioDatiSSD/ROD",
                        help="The input folder")

    args = parser.parse_args()
    path = args.path

    ids_file = open(path+"/testinstance_ids.txt", "r")
    map_ids = []
    for line in ids_file:
        map_ids.append(line.strip())
        print(line.strip())

    i = 0
    for classname in os.listdir(path):
        if not "testinstance_ids.txt" == classname and not "train" == classname and not "val" == classname:
            i += 1
            print(str(i) + " " + classname)

            src_dir = path+"/"+classname
            str_i = '%0*d' % (2, i+1)

            for instdir in os.listdir(src_dir):
                if str(instdir) in map_ids:
                    dest = path+"/val/"+str_i
                else:
                    dest = path+"/train/"+str_i

                src = src_dir + "/" + instdir
                print("\t " + src)

                if not os.path.exists(dest):
                    os.mkdir(dest)

                for image in os.listdir(src):
                    if "crop" in image and "depth" not in image and "mask" not in image:
                        copyfile(src+"/"+image, dest+"/"+image)
