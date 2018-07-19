import os
import argparse

dirs_path = {
    "rod1": "/home/fabio/robot_challenge/rod/split1/val",
    "arid": "/home/fabio/robot_challenge/arid/all",
    "rod2": "/home/fabio/robot_challenge/rod/split2/val",
    "rod3": "/home/fabio/robot_challenge/rod/split3/val"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to transform ARID into Image Folder')
    parser.add_argument('--dataset', type=str, default="arid",
                        help="The input dataset")

    args = parser.parse_args()
    dataset = args.dataset

    for n_class in sorted(os.listdir(dirs_path[dataset])):
        file = os.listdir(dirs_path[dataset]+"/"+n_class)[0]
        print(n_class + "\t" + file)
