import training
import OBC.networks
import argparse
import torch.nn as nn
import data_loader as dl
import math
import OBC.ClassificationMetric
from torchvision.datasets import ImageFolder

task_list = ["PE", "SC", "OC"]
folders = {
    "PE": '/home/fabio/robot_challenge/linemod',
    "SC": '/home/fabio/robot_challenge/NYU',
    "OC": '/home/fabio/robot_challenge/rod/split1'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Masked model for VDA challenge')
    # NAMING-PARAMETERS
    parser.add_argument('--folder', type=str, default=None,
                        help='Where to locate the imgs')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Whether to use a pretrained model.')
    parser.add_argument('--prefix', type=str, default='./models',
                        help='Where to store the checkpoints')
    parser.add_argument('--visdom', type=str, default="training",
                        help='Select the visdom environment.')
    # TRAINING PARAMETERS
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate to apply into training')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='The learning rate to apply into training')
    parser.add_argument('--bs', type=int, default=64,
                        help='The learning rate to apply into training')
    parser.add_argument('--step', type=int, default=40,
                        help='The learning rate to apply into training')
    parser.add_argument('--epochs', type=int, default=60,
                        help='How many epochs to use for training')
    # SETTING PARAMETERS
    parser.add_argument('--frozen', type=int, default=0,
                        help='Whether to use fine tuning (0 - DEF) or feature extractor (1).')
    parser.add_argument('--task', type=str, default='PE',
                        help='Which is the task to run')

    args = parser.parse_args()
    
    TEST = 0
    task = args.task
    if task not in task_list:
        raise(Exception("Please be sure to use available task"))
        
    folder = args.folder if args.folder else folders[task]
    
    if task == "PE":
        import PoseEstimation.PELoss as pel
        from PoseEstimation.LinemodDataset import LinemodDataset
        classes = 15 + 3
        cost_function = pel.PE3DLoss(classes - 3)
        metric = pel.PEMetric(classes - 3, math.radians(5))
        train_loader = dl.get_image_folder_loaders(folder + "/train", LinemodDataset, False, False, args.bs)
        test_loader = dl.get_image_folder_loaders(folder + "/val", LinemodDataset, False, False, args.bs)
    elif task == "OC":
        classes = 51
        cost_function = nn.CrossEntropyLoss()
        metric = OBC.ClassificationMetric.ClassificationMetric()
        # Image folder for train and val
        train_loader = dl.get_image_folder_loaders(folder + "/train", ImageFolder, False, True, args.bs)
        test_loader = dl.get_image_folder_loaders(folder + "/val", ImageFolder, False, False, args.bs)
    elif task == "SC":
        classes = 31
        cost_function = nn.CrossEntropyLoss()
        metric = OBC.ClassificationMetric.ClassificationMetric()
        # Image folder for train and val
        train_loader = dl.get_image_folder_loaders(folder + "/train", ImageFolder, False, True, args.bs)
        test_loader = dl.get_image_folder_loaders(folder + "/val", ImageFolder, False, False, args.bs)
    else:
        # never executed, needed only for remove warnings.
        raise(Exception("Error in parameters. It should be one between [SC,OC,PE]"))
    
    # basic network (will be changed according to te baseline)
    model = OBC.networks.resnet18(classes, args.pretrained)
    
    accuracy = 0
    if not TEST:
        accuracy = training.train(model, train_loader, test_loader, prefix=args.prefix, visdom_env=args.visdom,
                                  step=args.step, batch=args.bs, epochs=args.epochs, lr=args.lr, decay=args.decay,
                                  freeze=args.frozen, cost_function=cost_function, metric=metric)
    else:
        accuracy = training.test(model, test_loader, cost_function, metric)

    print("Final accuracy = " + str(accuracy))
