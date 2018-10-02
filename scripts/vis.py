import numpy as np
import visdom
import os
from numpy import genfromtxt

vis = visdom.Visdom()
path = "../log/"
for file in os.listdir(path):
# Define visualization environment
    vis.env = file[:-4]
    data = genfromtxt(os.path.join(path, file), delimiter=',')
    
    iters = data[0]
    losses_training = data[1]
    losses_test = data[2]
    accuracies_test = data[3]

    vis.line(
          X=np.array(iters),
          Y=np.array(losses_training),
          opts={
              'title' : ' Training Loss ',
              'xlabel': 'iterations',
              'ylabel': 'loss'},
          name='Training Loss ',
          win=0)
    vis.line(
          X=np.array(iters),
          Y=np.array(losses_test),
          opts={
              'title' : ' Validation Loss ',
              'xlabel': 'iterations',
              'ylabel': 'loss'},
          name='Validation Loss ',
          win=1)
    vis.line(
          X=np.array(iters),
          Y=np.array(accuracies_test),
          opts={
              'title' : ' Accuracy ',
              'xlabel': 'iterations',
              'ylabel': 'accuracy'},
          name='Validation Accuracy ',
          win=2)