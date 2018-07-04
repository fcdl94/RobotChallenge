import training
import torchvision
import argparse
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Masked model for VDA challenge')
    parser.add_argument('--net', type=str, default='piggyback',
                        help='Network that we want to train. Possible values: resnet, piggyback, quantized.')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Whether to use a pretrained model.')
    parser.add_argument('--prefix', type=str, default='./models',
                        help='Where to store the checkpoints')
    parser.add_argument('--bn', type=int, default=0,
                        help='Whether to tune Batch-Normalization layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning rate to apply into training')
    parser.add_argument('--scaling', type=int, default=1,
                        help='Whether to apply scaling as data-augmentation.')
    parser.add_argument('--frozen', type=int, default=0,
                        help='Whether to use fine tuning (0 - DEF) or feature extractor (1).')
    parser.add_argument('--epochs', type=int, default=60,
                        help='How many epochs to use for training')
    parser.add_argument('--test', type=int, default=0,
                        help='Whether it is only to test or also to train.')
    parser.add_argument('--output', type=str, default=None,
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
        accuracy = training.train(model, args.prefix, freeze=args.frozen,
                                  epochs=args.epochs, visdom_env=args.visdom, lr=args.lr)
    else:
        accuracy = training.test(model)

    print("Final accuracy = " + str(accuracy))
