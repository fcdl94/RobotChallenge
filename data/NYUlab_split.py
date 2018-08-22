import os
import argparse
from shutil import move

SCENES = [
    "bedroom",
    "kitchen",
    "living_room",
    "bathroom",
    "dining_room",
    "office",
    "home_office",
    "classroom",
    "bookstore",
    "other"
    # study student_lounge basement reception_room printer_room playroom office_kitchen laundry_room
    # indoor_balcony home_storage foyer exercise_room dinette conference_room computer_lab furniture_store cafe
]

if __name__ == "__main__":
    # Split the data-set, one image every 5 is in the validation set
    # I expect to have:
        # scenes.txt that is a list containing the scene pf the image (ordered)
        # train.txt that is the list of training samples (as number)
    parser = argparse.ArgumentParser(description='Script to split NYU')
    parser.add_argument('--path', type=str, default="/home/fabio/robot_challenge/NYUlab/data",
                        help="The input folder")
    
    args = parser.parse_args()
    path = args.path
    # GETTING FOLDER NAME AND CREATING PARENT FOLDERS
    # todo nei dati rimuovere lo split, basta fare una semplice move di tutto quello all'intero e cancellare train e val
    images = os.path.join(path, "images")
    depth = os.path.join(path, "depth")

    dest_image_dir_val = os.path.join(path, "val")
    dest_image_dir_train = os.path.join(path, "train")
    os.mkdir(dest_image_dir_val)
    os.mkdir(dest_image_dir_train)

    # GETTING META FILES
    train_path = os.path.join(path, "train.txt")
    scenes_path = os.path.join(path, "scenes.txt")
    train_file = open(train_path, "r")
    scenes_file = open(scenes_path, "r")
    
    scenes = [l.split(" ")[1][:-6] for l in scenes_file]
    train_list = [int(l) for l in train_file]

    # CREATE FOLDERS FOR EACH CLASS
    for classname in SCENES:
        os.mkdir(os.path.join(dest_image_dir_val, classname))
        os.mkdir(os.path.join(dest_image_dir_train, classname))
        
    # COPY IMAGE INTO CORRECT FOLDER
    for i in range(0, 1449):
        image_name = "{:08d}.jpg".format(i+1)
        depth_name = "{:08d}.png".format(i+1)
        
        if i in train_list:
            if scenes[i] in SCENES:
                dest_image = os.path.join(dest_image_dir_train, scenes[i])
            else:
                dest_image = os.path.join(dest_image_dir_train, "other")
        else:
            if scenes[i] in SCENES:
                dest_image = os.path.join(dest_image_dir_val, scenes[i])
            else:
                dest_image = os.path.join(dest_image_dir_val, "other")
                
        move(os.path.join(images, image_name), dest_image)
        move(os.path.join(depth, depth_name), dest_image)
