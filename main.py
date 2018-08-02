import training
import OBC.networks
import argparse
import torch.nn as nn
import data_loader as dl
import OBC.ClassificationMetric
from torchvision.datasets import ImageFolder

task_list = [
    "PE",
    "Classification"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Masked model for VDA challenge')
    parser.add_argument('--folder', type=str, default='/home/fabio/robot_challenge/rod/split1',
                        help='Where to locate the imgs')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Whether to use a pretrained model.')
    parser.add_argument('--prefix', type=str, default='./models',
                        help='Where to store the checkpoints')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The learning rate to apply into training')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='The learning rate to apply into training')
    parser.add_argument('--bs', type=int, default=64,
                        help='The learning rate to apply into training')
    parser.add_argument('--step', type=int, default=40,
                        help='The learning rate to apply into training')
    parser.add_argument('--frozen', type=int, default=0,
                        help='Whether to use fine tuning (0 - DEF) or feature extractor (1).')
    parser.add_argument('--epochs', type=int, default=60,
                        help='How many epochs to use for training')
    parser.add_argument('--test', type=int, default=0,
                        help='Whether it is only to test or also to train.')
    parser.add_argument('--visdom', type=str, default="training",
                        help='Select the visdom environment.')
    parser.add_argument('--task', type=str, default='PE',
                        help='Which is the task to run')

    args = parser.parse_args()
    
    task = args.task
    if task not in task_list:
        raise(Exception("Please be sure to use available task"))

    classes = 1
    if task == "PE":
        import PoseEstimation.PELoss as pel
        from PoseEstimation.LinemodDataset import LinemodDataset
        classes = 2 + 3
        cost_function = pel.PE3DLoss(2)
        metric = pel.PEMetric(2, 0.1)  # 0.04 means nearly 5 degrees
        train_loader = dl.get_image_folder_loaders(args.folder, LinemodDataset, False, True, args.bs)
        test_loader = dl.get_image_folder_loaders(args.folder, LinemodDataset, False, False, args.bs)
    elif task == "Classification":
        classes = 51
        cost_function = nn.CrossEntropyLoss()
        metric = OBC.ClassificationMetric.ClassificationMetric()
        # Image folder for train and val
        train_loader = dl.get_image_folder_loaders(args.folder + "/train", ImageFolder, False, True, args.bs)
        test_loader = dl.get_image_folder_loaders(args.folder + "/val", ImageFolder, False, False, args.bs)
        
    model = OBC.networks.resnet18(classes, args.pretrained)
    
    accuracy = 0
    if not args.test:
        # (model, prefix, freeze=False, lr=0.001, momentum=0.9, epochs=EPOCHS, visdom_env="robotROD"):
        accuracy = training.train(model, train_loader, test_loader, args.prefix, freeze=args.frozen, step=args.step, batch=args.bs,
                                  epochs=args.epochs, visdom_env=args.visdom, lr=args.lr, decay=args.decay,
                                  cost_function_p=cost_function, metric=metric)
    else:
        accuracy = training.test(model, args.folder, cost_function, metric)

    print("Final accuracy = " + str(accuracy))


