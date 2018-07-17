import os
import argparse
from shutil import copyfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to transform ARID into Image Folder')
    parser.add_argument('--path', type=str, default="/home/fabio/robot_challenge/arid",
                        help="The input folder")
    parser.add_argument('--index', type=int, default=1,
                        help="The instance to be considered for test")

    args = parser.parse_args()
    path = args.path
    if args.index and 0 > args.index <= 3:
        test_index = args.index
    else:
        test_index = 1

    i = 0
    test_index -= 1

    for classname in os.listdir(path):
        if not "train" == classname and not "val" == classname:
            i += 1
            print(str(i) + " " + classname)

            src_dir = path+"/"+classname
            str_i = '%0*d' % (2, i)

            for j, instdir in enumerate(os.listdir(src_dir)):
                if j == test_index:
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
