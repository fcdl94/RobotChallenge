import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms, datasets
from datetime import datetime
import numpy as np
from DIAL.entropy_loss import EntropyLoss
from DIAL.DoubleDataset import DoubleDataset
import visdom
import itertools

# Training settings
PATH_TO_DATASETS = '/home/lab2atpolito/FabioDatiSSD/ROD'
BATCH_SIZE = 64
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


def train(model, folder_source, folder_target, freeze=False, lr=0.001, momentum=0.9, epochs=EPOCHS, visdom_env="Auto_DIAL_PyTorch",
          decay=10e-5, step=STEP, batch=BATCH_SIZE):
    # Define visualization environment
    vis.env = visdom_env
    global BATCH_SIZE
    global STEP
    
    BATCH_SIZE = batch
    STEP = step
    
    # data pre-processing
    workers = WORKERS if cuda else 0
    data_transform = get_data_transform(True, False)

    # source domain images
    s_dataset = datasets.ImageFolder(root=folder_source + '/train', transform=data_transform)
    t_dataset = datasets.ImageFolder(root=folder_target + '/train', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(DoubleDataset(s_dataset, t_dataset), batch_size=batch, shuffle=True, num_workers=workers)

    # Build the test loader
    # (note that more complex data transforms can be used to provide better performances e.g. 10 crops)
    data_transform = get_data_transform(False, False)

    s_dataset = datasets.ImageFolder(root=folder_source + '/val', transform=data_transform)
    t_dataset = datasets.ImageFolder(root=folder_target + '/train', transform=data_transform)
    s_test_loader = torch.utils.data.DataLoader(s_dataset, batch_size=batch, shuffle=False, num_workers=workers)
    d_test_loader = torch.utils.data.DataLoader(t_dataset, batch_size=batch, shuffle=False, num_workers=workers)
    
    params_to_optim = list(filter(lambda p: p.requires_grad, model.parameters()))

    # set optimizer and scheduler
    optimizer = optim.SGD(params_to_optim, lr=lr, momentum=momentum, weight_decay=decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step)

    # prepare for training
    if cuda:
        model = model.cuda()

    # Initialize the lists needed for visualization, plus window offset for the graphs
    iters = []
    source_losses_training = []
    target_losses_training = []
    losses_test = []
    accuracies_test = []

    # result = test_epoch(model, -1, s_test_loader, d_test_loader)
    # print('Test at start time\tAccuracyTarget: {:.6f}\tAccuracySource: {:.6f}'.format(
    #      result[1], result[0]))
    
    # perform training epochs time
    best_accuracy = -1
    #val_epoch_min = -1
    for epoch in range(1, epochs + 1):

        scheduler.step()
        print(str(epoch) + "-lr: " + str(optimizer.state_dict()["param_groups"][0]["lr"]))

        loss_epoch = train_epoch(model, epoch, train_loader, optimizer)
        result = test_epoch(model, epoch, s_test_loader, d_test_loader)

        accuracies_test.append(result[0])
        losses_test.append(result[1])
        source_losses_training.append(loss_epoch[0])
        target_losses_training.append(loss_epoch[1])
        iters.append(epoch)

        print('Train Epoch: {} \tSourceTrainLoss: {:.6f} \tTestTrainLoss: {:.6f} \tAccuracySource: {:.6f}'
              '\tAccuracyTarget: {:.6f}'.format(
            epoch, loss_epoch[0], loss_epoch[1], result[0], result[1]))

        # Print results
        vis.line(
            X=np.array(iters),
            Y=np.array(source_losses_training),
            opts={
                'title': ' Source Training Loss ',
                'xlabel': 'iterations',
                'ylabel': 'loss'},
            name='Source Training Loss ',
            win=0)
        vis.line(
            X=np.array(iters),
            Y=np.array(target_losses_training),
            opts={
                'title': ' Target Training Loss ',
                'xlabel': 'iterations',
                'ylabel': 'loss'},
            name='Target Training Loss ',
            win=1)
        vis.line(
            X=np.array(iters),
            Y=np.array(losses_test),
            opts={
                'title': ' Accuracy Target ',
                'xlabel': 'iterations',
                'ylabel': 'accuracy'},
            name='Target Accuracy ',
            win=2)
        vis.line(
            X=np.array(iters),
            Y=np.array(accuracies_test),
            opts={
                'title': ' Accuracy Source ',
                'xlabel': 'iterations',
                'ylabel': 'accuracy'},
            name='Source Accuracy ',
            win=3)

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
def train_epoch(model, epoch, data_loader, optimizers):
    # Set the model in training mode
    model.train()

    source_cost = nn.CrossEntropyLoss()
    #target_cost = nn.CrossEntropyLoss()
    target_cost = EntropyLoss()
    
    print("Starting time of Epoch " + str(epoch) + ": " + str(datetime.now().time()))

    # Init holders
    source_losses = 0
    target_losses = 0
    batch_idx = 1
    
    # Perform the training procedure
    for s_data, t_data in data_loader:
    
        (source_data, source_target) = s_data
        (target_data, target_target) = t_data

        # Reset the optimizers
        optimizers.zero_grad()
        
        # DO that for SOURCE
        # Move the variables to GPU
        data, target = source_data, source_target
        if cuda:
            data, target = data.cuda(), target.cuda()
        # Indicate to use the source DA
        model.set_domain(True)
        # Process input
        output = model(data)
        # Compute loss and gradients
        source_loss = source_cost(output, target)
        
        # DO that for TARGET
        # Move the variables to GPU
        data = target_data
        if cuda:
            data = data.cuda()
        # Indicate to use the target DA
        model.set_domain(False)
        # Process input
        output = model(data)
        # Compute loss and gradients
        target_loss = target_cost(output)

        # Backward and update
        loss = source_loss + LAMBDA * target_loss
        loss.backward()

        # if batch_idx % LOG_INTERVAL == 0:
        #    print("BN source grad" + str(model.bn1.bn_source.weight.grad))
        #    print("BN target grad" + str(model.bn1.bn_target.weight.grad))

        optimizers.step()

        # Check for log and update holders
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{:6d}/{:6d} ({:2.0f}%)]\tLoss: {:.6f}\tSource Loss: {:.6f}\tTarget Loss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE*2, len(data_loader.dataset)*2,
                       100. * batch_idx / len(data_loader), loss.item(), source_loss.item(), target_loss.item()))

        source_losses += source_loss.item()
        target_losses += target_loss.item()
        batch_idx += 1
    current = batch_idx-1
    return [source_losses/current,   target_losses/current]


def test_epoch(model, epoch, s_loader, t_loader):
    # Put the model in eval mode
    model.eval()
    torch.set_grad_enabled(False)
    # Init holders
    s_correct = 0
    t_correct = 0
    
    # Perform the evaluation procedure
    for data, target in s_loader:
        
        if cuda:
            data, target = data.cuda(), target.cuda()

        model.set_domain(True)

        output = model(data)

        pred = torch.max(output, 1)[1]  # get the index of the max log-probability
        s_correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # Check if the prediction is correct

    # Perform the evaluation procedure
    for data, target in t_loader:
        # Reset and compute for target distribution
        if cuda:
            data, target = data.cuda(), target.cuda()
    
        model.set_domain(False)
        output = model(data)
    
        pred = torch.max(output, 1)[1]  # get the index of the max log-probability
        t_correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # Check if the prediction is correct

    source_accuracy = 100. * float(s_correct) / (len(s_loader.dataset))
    target_accuracy = 100. * float(t_correct) / (len(t_loader.dataset))
    
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

