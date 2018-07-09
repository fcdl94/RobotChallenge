import training
import torchvision
import argparse
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Masked model for VDA challenge')
    parser.add_argument('--folder', type=str, default='/home/lab2atpolito/FabioDatiSSD/ROD',
                        help='Where to locate the imgs')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Whether to use a pretrained model.')
    parser.add_argument('--prefix', type=str, default='./models',
                        help='Where to store the checkpoints')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning rate to apply into training')
    parser.add_argument('--decay', type=float, default=10e-5,
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

    args = parser.parse_args()

    model = torchvision.models.resnet18(True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 51)

    accuracy = 0
    if not args.test:
        # (model, prefix, freeze=False, lr=0.001, momentum=0.9, epochs=EPOCHS, visdom_env="robotROD"):
        accuracy = training.train(model, args.folder, args.prefix, freeze=args.frozen, step=args.step,
                                  epochs=args.epochs, visdom_env=args.visdom, lr=args.lr, decay=args.decay)
    else:
        accuracy = training.test(model)

    print("Final accuracy = " + str(accuracy))
