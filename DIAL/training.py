import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms, datasets
from datetime import datetime
import numpy as np
from DIAL.entropy_loss import EntropyLoss
import visdom
import itertools

# Training settings
PATH_TO_DATASETS = '/home/lab2atpolito/FabioDatiSSD/ROD'
BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
EPOCHS = 60
STEP = 40
NO_CUDA = False
IMAGE_CROP = 224
LOG_INTERVAL = 10
WORKERS = 8
LAMBDA = 0.1

# image normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
#IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_STD = [1, 1, 1]

# Initialize visualization tool
vis = visdom.Visdom()

# Check for CUDA usage
cuda = not NO_CUDA and torch.cuda.is_available()


def train(model, folder, prefix, freeze=False, lr=0.001, momentum=0.9, epochs=EPOCHS, visdom_env="Auto_DIAL_PyTorch",
          decay=10e-5, step=STEP, batch=BATCH_SIZE):
    # Define visualization environment
    vis.env = visdom_env

    # data pre-processing
    workers = WORKERS if cuda else 0
    data_transform = get_data_transform(True, False)

    # source domain images
    dataset = datasets.ImageFolder(root=folder + '/train', transform=data_transform)
    source_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers)

    # Build the test loader
    # (note that more complex data transforms can be used to provide better performances e.g. 10 crops)
    data_transform = get_data_transform(False, False)

    dataset = datasets.ImageFolder(root=folder + '/val', transform=data_transform)
    source_test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers)

    # target domain dataset - without labels - it must be used whole for test
    data_transform = get_data_transform(True, False)

    dataset = datasets.ImageFolder(root=folder + '/train', transform=data_transform)
    target_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers)

    data_transform = get_data_transform(False, False)

    dataset = datasets.ImageFolder(root=folder + '/val', transform=data_transform)
    target_test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers)

    # If feature extractor free all the network except fc
    if freeze:
        for name, par in model.named_parameters():
            if "fc" not in name:
                par.requires_grad = False

    params_to_optim = list(filter(lambda p: p.requires_grad, model.parameters()))

    # set optimizer and scheduler
    optimizer = optim.SGD(params_to_optim, lr=lr, momentum=momentum, weight_decay=decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step)

    # prepare for training
    if cuda:
        model = model.cuda()

    # Initialize the lists needed for visualization, plus window offset for the graphs
    iters = []
    losses_training = []
    losses_test = []
    accuracies_test = []

    # perform training epochs time
    best_accuracy = -1
    #val_epoch_min = -1
    for epoch in range(1, epochs + 1):

        scheduler.step()
        print(str(epoch) + "-lr: " + str(optimizer.state_dict()["param_groups"][0]["lr"]))

        loss_epoch = train_epoch(model, epoch, source_train_loader, target_train_loader, optimizer, not freeze)
        result = test_epoch(model, source_test_loader, target_test_loader)

        accuracies_test.append(result[0])
        losses_test.append(result[1])
        losses_training.append(loss_epoch)
        iters.append(epoch)

        print('Train Epoch: {} \tTrainLoss: {:.6f} \tAccuracyTarget: {:.6f}\tAccuracySource: {:.6f}'.format(
            epoch, loss_epoch, result[1], result[0]))

        # Print results
        vis.line(
            X=np.array(iters),
            Y=np.array(losses_training),
            opts={
                'title': ' Training Loss ',
                'xlabel': 'iterations',
                'ylabel': 'loss'},
            name='Training Loss ',
            win=0)
        vis.line(
            X=np.array(iters),
            Y=np.array(losses_test),
            opts={
                'title': ' Accuracy Target ',
                'xlabel': 'iterations',
                'ylabel': 'accuracy'},
            name='Target Accuracy ',
            win=1 )
        vis.line(
            X=np.array(iters),
            Y=np.array(accuracies_test),
            opts={
                'title': ' Accuracy Source ',
                'xlabel': 'iterations',
                'ylabel': 'accuracy'},
            name='Source Accuracy ',
            win=2 )

        if best_accuracy < result[0]:
            best_accuracy = result[0]

        # Save the model
        #if result[1] <= val_epoch_min or val_epoch_min == -1:
        #    val_epoch_min = loss_epoch
        #    torch.save({
        #        'epoch': epoch,
        #        'state_dict': model.state_dict(),
        #        'optimizer': optimizer.state_dict()
        #    }, prefix + ".pth")

    return best_accuracy


def test(model):
    # Training steps:
    # Preprocessing (cropping, hor-flipping, resizing) and get data
    # Initialize data processing threads
    workers = WORKERS if cuda else 0

    # Build the test loader
    # (note that more complex data transforms can be used to provide better performances e.g. 10 crops)
    data_transform = get_data_transform(False, True)

    dataset = datasets.ImageFolder(root=PATH_TO_DATASETS + '/val', transform=data_transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)

    # set loss function
    cost_function = nn.CrossEntropyLoss()

    if cuda:
        model = model.cuda()

    result = test_epoch(model, test_loader, cost_function)
    print('Test \tTestLoss: {:.6f}\tAccuracyTest: {:.6f}'.format(
           result[1], result[0]))

    return result[0]


# Perform a single training epoch
def train_epoch(model, epoch, source_loader, target_loader, optimizers, bn=False):
    # Set the model in training mode
    model.train()

    source_cost = nn.CrossEntropyLoss()
    target_cost = EntropyLoss()
    
    print("Starting time of Epoch " + str(epoch) + ": " + str(datetime.now().time()))
    # If BN parameters must be frozen, freeze them
    if not bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.eval()

    # Init holders
    losses = 0
    current = 0

    # Perform the training procedure
    for batch_idx, ((source_data, source_target), (target_data, _ )) in enumerate(itertools.product(source_loader, target_loader)):

        # DO that for source
        # Move the variables to GPU
        data, target = source_data, source_target
        
        if cuda:
            data, target = data.cuda(), target.cuda()

        model.set_domain(True)  # it indicates to use the source DA
        # Reset the optimizers
        optimizers.zero_grad()

        # Process input
        output = model(data)

        # Compute loss and gradients
        source_loss = source_cost(output, target)
        
        # DO that for target
        data = target_data
        if cuda:
            data = data.cuda()  # we don't use labels for target
        
        model.set_domain(False)  # it indicates to use target DA
        # Reset the optimizers
        optimizers.zero_grad()

        # Process input
        output = model(data)

        # Compute loss and gradients
        target_loss = target_cost(output)

        loss = source_loss + LAMBDA * target_loss
        
        loss.backward()

        # Update parameters
        optimizers.step()

        # Check for log and update holders
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{:4d}/{:4d}-{:4d} ({:2.0f}%)]\tAvgLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(source_loader.dataset), len(target_loader.dataset),
                       100. * batch_idx / len(source_loader), loss.item() / BATCH_SIZE))

        losses += loss.item()
        current += 1

    return losses / current


def test_epoch(model, source_loader, target_loader):
    # Put the model in eval mode
    model.eval()
    torch.set_grad_enabled(False)
    # Init holders
    correct = 0

    # Perform the evaluation procedure
    for data, target in source_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        pred = torch.max(output, 1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # Check if the prediction is correct

    # Compute accuracy
    source_accuracy = 100. * float(correct) / (len(source_loader.dataset))

    # Reset and compute for target distribution
    correct = 0
    # Perform the evaluation procedure
    for data, target in target_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
    
        output = model(data)
    
        pred = torch.max(output, 1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # Check if the prediction is correct

    target_accuracy = 100. * float(correct) / (len(target_loader.dataset))
    
    results = [source_accuracy, target_accuracy]

    torch.set_grad_enabled(True)
    return results


def get_data_transform(mirror, scaling):
    # Create Data loader w.r.t. chosen transformations
    if mirror:
        if scaling:
            data_transform = transforms.Compose([
                transforms.RandomResizedCrop(IMAGE_CROP, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            data_transform = transforms.Compose([
                transforms.Resize((IMAGE_CROP, IMAGE_CROP)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
    else:
        if scaling:
            data_transform = transforms.Compose([
                #transforms.Resize((IMAGE_CROP, IMAGE_CROP)),
                transforms.RandomResizedCrop(IMAGE_CROP, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            data_transform = transforms.Compose([
                transforms.Resize((IMAGE_CROP, IMAGE_CROP)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
    return data_transform

