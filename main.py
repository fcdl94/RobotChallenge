import training
import OBC.networks
import argparse
import torch.nn as nn
import data_loader as dl
import OBC.ClassificationMetric
import Depth.RGBDNet as RGBDNet
import par_sets as ps
from datetime import date
import Piggyback.networks as pbnet
import Rebuffi.networks as rbnet
import CombinedNet.networks as cbnet

task_list = ["OC", "PE", "SC", "TE"]
folders = {
    "PE": '/home/fabioc/dataset/linemod',
    "SC": '/home/fabioc/dataset/nyu',
    "OC": '/home/fabioc/dataset/rod',
    "DEBUG": '/home/fcdl/Develop/Data/sample',
}
network_list = ["resnet", "piggyback", "quantized", "serial", "parallel", "combined"]

classes_list = {
    "OC": 51,
    "PE": 19,
    "SC": 10
}


def parse_order(order):
    order = str(order)
    if len(order) == 2:
        order = '0' + order
    return [int(i) for i in order]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Masked model for VDA challenge')
    # NAMING-PARAMETERS
    parser.add_argument('--folder', type=str, default=None,
                        help='Where to locate the imgs')
    parser.add_argument('-p', '--pretrained', type=str, default=None,
                        help='Whether to use a pretrained model.')
    parser.add_argument('--prefix', type=str, default='noName',
                        help='Where to store the checkpoints')
    parser.add_argument('-v', '--visdom', type=str, default="training",
                        help='Select the visdom environment.')
    parser.add_argument('--name', type=str, default=None,
                        help='If this is given, visdom and prefix will be called as this.')
    # TRAINING PARAMETERS
    parser.add_argument('--lr', type=float, default=None,
                        help='The learning rate to apply into training')
    parser.add_argument('--adamlr', type=float, default=None,
                        help='The ADAM learning rate to apply into training if piggyback on OC '
                             '(it is multiplied by 5 in PE, by 10 in SC')
    parser.add_argument('--decay', type=float, default=None,
                        help='The learning rate to apply into training')
    parser.add_argument('--bs', type=int, default=None,
                        help='The learning rate to apply into training')
    parser.add_argument('--step', type=int, default=None,
                        help='The learning rate to apply into training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='How many epochs to use for training')
    parser.add_argument('--set', type=str, default="rod",
                        help='The parameter set')
    # SETTING PARAMETERS
    parser.add_argument('--frozen', type=bool, default=False,
                        help='Whether to use fine tuning (0 - DEF) or feature extractor (1).')
    parser.add_argument('-t', '--task', type=str, default='PE',
                        help='Which is the task to run')
    parser.add_argument('-n', '--network', type=str, default='resnet',
                        help='Which is the network to use')
    parser.add_argument("--depth", type=int, default=0,
                        help="if this is true, depth will be used.")
    parser.add_argument("--rgb", type=int, default=1,
                        help="if this is true, rgb will be used.")
    parser.add_argument("--order", type=int, default=12,
                        help="if this is true, rgb will be used.")

    args = parser.parse_args()
    
    TEST = 0
    task = args.task
    if task not in task_list:
        raise(Exception("Please be sure to use available task"))
    
    if args.name:
        visdom = args.name
        prefix = args.name
    else:
        visdom = args.visdom
        prefix = args.prefix
    
    folder = args.folder if args.folder else folders[task]
    
    depth = args.depth
    rgb = args.rgb

    # PLEASE READ THIS. ADAM is multipled by 5 in PE and by 10 in SC (given experiments, it's the best way)
    par_set = ps.get_parameter_set(args.set)
    step = args.step if args.step else par_set["step"]
    batch = args.bs if args.bs else par_set["bs"]
    epochs = args.epochs if args.epochs else par_set["epochs"]
    lr = args.lr if args.lr else par_set["lr"]
    adamlr = args.adamlr if args.adamlr else par_set["adamlr"]
    decay = args.decay if args.decay else par_set["decay"]
    
    if task == "OC":
        from Depth.RODDataset import RODDataset
        cost_function = nn.CrossEntropyLoss()
        metric = OBC.ClassificationMetric.ClassificationMetric()
        # Image folder for train and val
        train_loader = dl.get_image_folder_loaders(folder + "/train", RODDataset, "SC", batch, rgb, depth)
        test_loader = dl.get_image_folder_loaders(folder + "/val", RODDataset, "NO", batch, rgb, depth)
        index = 0
    elif task == "PE":
        import PoseEstimation.PELoss as pel
        from PoseEstimation.LinemodDataset import LinemodDataset
        cost_function = pel.PE3DLoss(classes_list["PE"] - 4)
        metric = pel.PEMetric(classes_list["PE"] - 4, threshold=20)
        # revert here. train / val not sample
        train_loader = dl.get_image_folder_loaders(folder + "/train", LinemodDataset, "NO", batch, rgb, depth)
        test_loader = dl.get_image_folder_loaders(folder + "/val", LinemodDataset, "NO", batch, rgb, depth)
        index = 1
        adamlr *= 5
    elif task == "SC":
        from Depth.NYUDataset import NYUDataset
        cost_function = nn.CrossEntropyLoss()
        metric = OBC.ClassificationMetric.ClassificationMetric()
        # Image folder for train and val
        train_loader = dl.get_image_folder_loaders(folder + "/train", NYUDataset, "SM", batch, rgb, depth)
        test_loader = dl.get_image_folder_loaders(folder + "/val", NYUDataset, "NO", batch, rgb, depth)
        index = 2
        adamlr *= 10
    else:
        # never executed, needed only for remove warnings.
        raise(Exception("Error in parameters. Task should be one between " + str(task_list)))
    
    # basic network (will be changed according to te baseline)
    if args.network == network_list[0]:  # resnet
        if depth and rgb:
            model = RGBDNet.double_resnet18(classes_list[task])
        else:
            model = OBC.networks.resnet18(classes_list[task], args.pretrained)
    elif args.network == network_list[1]:  # piggyback
        if depth and rgb:
            model = RGBDNet.double_piggyback18(classes_list.values(), index, args.pretrained)
        else:
            model = pbnet.piggyback_net18(classes_list.values(), pre_imagenet=True, pretrained=args.pretrained)
            model.set_index(index)
    elif args.network == network_list[2]:  # quantized
        if depth and rgb:
            model = RGBDNet.double_quantized18(classes_list.values(), index, args.pretrained)
        else:
            model = pbnet.quantized_net18(classes_list.values(), pre_imagenet=True, pretrained=args.pretrained)
            model.set_index(index)
    elif args.network == network_list[3]:  # serial
        if depth and rgb:
            model = RGBDNet.double_serial18(classes_list.values(), index, args.pretrained)
        else:
            model = rbnet.rebuffi_net18(classes_list.values(), pre_imagenet=True, pretrained=args.pretrained)
            model.set_index(index)
    elif args.network == network_list[4]:  # parallel
        if depth and rgb:
            model = RGBDNet.double_parallel18(classes_list.values(), index, args.pretrained)
        else:
            model = rbnet.rebuffi_net18(classes_list.values(), serie=False, pre_imagenet=True, pretrained=args.pretrained)
            model.set_index(index)
    elif args.network == network_list[5]:  # combined
        order = parse_order(args.order)
        if depth and rgb:
            model = RGBDNet.double_combined18(classes_list.values(), index, order, args.pretrained)
        else:
            model = cbnet.combined_net18(classes_list.values(), pre_imagenet=True, pretrained=args.pretrained,
                                         order=order)
            model.set_index(index)
    else:
        raise(Exception("Error in parameters. Network should be one between " + str(network_list)))
    
    if not TEST:
        accuracy = training.train(args.network, model, train_loader, test_loader, prefix=prefix, visdom_env=visdom,
                                  step=step, batch=batch, epochs=epochs, lr=lr, decay=decay, adamlr=adamlr, momentum=0.9,
                                  freeze=args.frozen, cost_function=cost_function, metric=metric)
    else:
        accuracy = training.test(model, test_loader, cost_function, metric)

    print("Saving results in RESULTS.txt")
    
    out = open("RESULTS.txt", "a")
    output = {
        "date"   : str(date.today()),
        "name"   : visdom,
        "task"   : args.task,
        "net"    : args.network,
        "rgb"    : rgb,
        "depth"  : depth,
        "epochs" : epochs,
        "adamlr" : adamlr,
        "lr"     : lr,
        "max_acc": "{:.2f}".format(accuracy[0]),
        "end_acc": "{:.2f}".format(accuracy[1])
    }
    print(str(output))
    out.write(str(output) + "\n")
    print("Result written")
    out.close()

