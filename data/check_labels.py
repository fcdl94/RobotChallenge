import os
import argparse

dirs_path = {
    "rod1": "/home/fabio/robot_challenge/split1",
    "arid": "/home/fabio/robot_challenge/arid"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to transform ARID into Image Folder')
    parser.add_argument('--dataset', type=str, default="arid",
                        help="The input dataset")

    args = parser.parse_args()
    dataset = args.dataset

    for n_class in os.listdir(dirs_path[dataset]):
        file = next(os.listdir(dirs_path[dataset]+"/"+n_class))
        print(n_class + "\t" + file)
