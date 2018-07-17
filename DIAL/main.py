import DIAL.training
import DIAL.networks
import torchvision
import argparse
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Masked model for VDA challenge')
    parser.add_argument('--source', type=str, default='/home/fabio/robot_challenge/rod/split1',
                        help='Where to locate the source imgs')
    parser.add_argument('--source', type=str, default='/home/fabio/robot_challenge/arid',
                        help='Where to locate the target imgs')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Whether to use a pretrained model.')
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
    parser.add_argument('--epochs', type=int, default=10,
                        help='How many epochs to use for training')
    parser.add_argument('--visdom', type=str, default="DIAL-PYTORCH_ROD-ARID",
                        help='Select the visdom environment.')

    args = parser.parse_args()

    model = DIAL.networks.resnet18(True)
    # adapt to ROD/ARID
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 51)

    accuracy = 0
    if not args.test:
        # (model, prefix, freeze=False, lr=0.001, momentum=0.9, epochs=EPOCHS, visdom_env="robotROD"):
        accuracy = DIAL.training.train(model, args.source, args.target, freeze=args.frozen, step=args.step, batch=args.bs,
                                  epochs=args.epochs, visdom_env=args.visdom, lr=args.lr, decay=args.decay)
    else:
        accuracy = DIAL.training.test(model)

    print("Final accuracy = " + str(accuracy))
