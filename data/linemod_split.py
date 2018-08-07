import os
import argparse
from shutil import copyfile

if __name__ == "__main__":
    # Split following the index, one image every 5 is in the validation set
    parser = argparse.ArgumentParser(description='Script to split Linemod')
    parser.add_argument('--path', type=str, default="/home/fabio/robot_challenge/linemod",
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

            while os.path.isfile(os.path.join(src_dir, "color" + str(index) + ".jpg")):
    
                path_img = os.path.join(src_dir, "color" + str(index) + ".jpg")
                path_rot = os.path.join(src_dir, "rot" + str(index) + ".rot")
                
                if (index+1) % 5 == 0:
                    # put into validation
                    dest_path_img = os.path.join(dest_dir_val,  "color" + str(index) + ".jpg")
                    dest_path_rot = os.path.join(dest_dir_val,  "color" + str(index) + ".rot")
                else:
                    # put into training
                    dest_path_img = os.path.join(dest_dir_train,  "color" + str(index) + ".jpg")
                    dest_path_rot = os.path.join(dest_dir_train,  "color" + str(index) + ".rot")
                    
                copyfile(path_img, dest_path_img)
                copyfile(path_rot, dest_path_rot)
                
                index += 1
