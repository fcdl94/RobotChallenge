import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from datetime import datetime
import numpy as np
import visdom
from OBC.ClassificationMetric import ClassificationMetric
from Piggyback.multiple_optim import MultipleOptimizer

# Training settings
EPOCHS = 60
STEP = 40
NO_CUDA = False
LOG_INTERVAL = 10

# Initialize visualization tool
vis = visdom.Visdom()

# Check for CUDA usage
cuda = not NO_CUDA and torch.cuda.is_available()


def train(network, model, train_loader, test_loader, freeze=False, prefix="checkpoint", visdom_env="robotROD",
          epochs=EPOCHS, step=STEP, lr=0.001, momentum=0.9, decay=10e-5, batch=32, adamlr=0.0001,
          cost_function=nn.CrossEntropyLoss(), metric=ClassificationMetric()):
    
    # Define visualization environment
    vis.env = visdom_env
    
    global BATCH_SIZE
    BATCH_SIZE = batch

    # If feature extractor free all the network except fc
    if freeze:
        for name, par in model.named_parameters():
            if "fc" not in name:
                par.requires_grad = False
    
    # Set optimizer and scheduler
    if not network == "piggyback":
        params_to_optim = list(filter(lambda p: p.requires_grad, model.parameters()))
        # set optimizer and scheduler
        optimizer = optim.SGD(params_to_optim, lr=lr, momentum=momentum, weight_decay=decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step)
    else:
        ignored_params = list(map(id, model.fc.parameters()))
        base_params = list(filter(lambda p: (id(p) not in ignored_params) and p.requires_grad, model.parameters()))
        fc_params = list(filter(lambda p: p.requires_grad, model.fc.parameters()))
        # set optimizer
        if len(base_params) == 0:
            optimizer_b = optim.SGD(fc_params, lr=lr, momentum=momentum, weight_decay=decay)
            scheduler_b = optim.lr_scheduler.StepLR(optimizer_b, step)
            scheduler = MultipleOptimizer(scheduler_b)
            optimizer = MultipleOptimizer(optimizer_b)
        else:
            optimizer_a = optim.Adam(base_params, lr=adamlr, weight_decay=decay)
            optimizer_b = optim.SGD(fc_params, lr=lr, momentum=momentum, weight_decay=decay)
            scheduler_a = optim.lr_scheduler.StepLR(optimizer_a, step)
            scheduler_b = optim.lr_scheduler.StepLR(optimizer_b, step)
            scheduler = MultipleOptimizer(scheduler_a, scheduler_b)
            optimizer = MultipleOptimizer(optimizer_a, optimizer_b)
    
    # Consider this as a sanity check
    for name, par in model.named_parameters():
        print(name + " requires_grad: " + str(par.requires_grad))
        
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
    val_epoch_min = -1
    for epoch in range(1, epochs + 1):

        scheduler.step()
        print(str(epoch) + "-lr: " + str(optimizer.state_dict()["param_groups"][0]["lr"]))

        # NOTE: loss function is a parameter as well as the metric function
        loss_epoch = train_epoch(model, epoch, train_loader, optimizer, cost_function)
        result = test_epoch(model, test_loader, cost_function, metric)

        accuracies_test.append(result[0])
        losses_test.append(result[1])
        losses_training.append(loss_epoch)
        iters.append(epoch)

        print('Train Epoch: {} \tTrainLoss: {:.6f} \tTestLoss: {:.6f}\tAccuracyTest: {:.6f}'.format(
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
                'title': ' Validation Loss ',
                'xlabel': 'iterations',
                'ylabel': 'loss'},
            name='Validation Loss ',
            win=1 )
        vis.line(
            X=np.array(iters),
            Y=np.array(accuracies_test),
            opts={
                'title': ' Accuracy ',
                'xlabel': 'iterations',
                'ylabel': 'accuracy'},
            name='Validation Accuracy ',
            win=2 )

        if best_accuracy < result[0]:
            best_accuracy = result[0]

        # Save the model
        if result[1] <= val_epoch_min or val_epoch_min == -1:
            val_epoch_min = loss_epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, "models/" + prefix + ".pth")

    return [best_accuracy, result[0]]


def test(model, test_loader, cost_function, metric):

    # set loss function in parameter as the metric

    if cuda:
        model = model.cuda()

    result = test_epoch(model, test_loader, cost_function, metric)
    print('Test \tTestLoss: {:.6f}\tAccuracyTest: {:.6f}'.format(
           result[1], result[0]))

    return result[0]


# Perform a single training epoch
def train_epoch(model, epoch, train_loader, optimizers, cost_function):
    # Set the model in training mode
    model.train()

    print("Starting time of Epoch " + str(epoch) + ": " + str(datetime.now().time()))

    # Init holders
    losses = 0

    # Perform the training procedure
    for batch_idx, (data, target) in enumerate(train_loader):

        # Move the variables to GPU
        if cuda:
            data, target = data.cuda(), target.cuda()

        # Reset the optimizers
        optimizers.zero_grad()

        # Process input
        output = model(data)

        # Compute loss and gradients
        loss = cost_function(output, target)
        loss.backward()

        # Update parameters
        optimizers.step()

        # Check for log and update holders
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{:4d}/{:4d} ({:2.0f}%)]\tAvgLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        losses += loss.item()

    return losses / len(train_loader)


def test_epoch(model, test_loader, cost_function, metric):
    # Put the model in eval mode
    model.eval()
    torch.set_grad_enabled(False)
    # Init holders
    test_loss = 0
    correct = 0

    # Perform the evaluation procedure
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        # Update holders
        test_loss += cost_function(output, target).item()  # sum up batch loss
        correct += metric(output, target)

    # Compute accuracy and loss
    total_loss = test_loss / len(test_loader)
    accuracy = 100. * float(correct) / (len(test_loader.dataset))

    results = [accuracy, total_loss]

    torch.set_grad_enabled(True)
    return results
