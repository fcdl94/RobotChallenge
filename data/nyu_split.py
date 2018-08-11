import os
import argparse
from shutil import copyfile

if __name__ == "__main__":
    # Split the data-set, one image every 5 is in the validation set
    parser = argparse.ArgumentParser(description='Script to split NYU')
    parser.add_argument('--path', type=str, default="/home/fabio/robot_challenge/NYU",
                        help="The input folder")

    args = parser.parse_args()
    path = args.path

    os.mkdir(path + "/val")
    os.mkdir(path + "/train")
    
    for classname in sorted(os.listdir(path)):
        if not "train" == classname and not "val" == classname:
            index = 0

            src_dir = os.path.join(path, classname)
            dest_dir_val = os.path.join(path, "val", classname)
            dest_dir_train = os.path.join(path, "train", classname)

            os.mkdir(dest_dir_val)
            os.mkdir(dest_dir_train)

            for i, file in enumerate(os.listdir(src_dir)):
                if index % 2 == 0:
                    
                    path_img = os.path.join(src_dir, file)
                
                    if (index+1) % 10 == 0:
                        # put into validation
                        dest_path_img = os.path.join(dest_dir_val,  file)
                    else:
                        # put into training
                        dest_path_img = os.path.join(dest_dir_train,  file)
                    
                    copyfile(path_img, dest_path_img)
