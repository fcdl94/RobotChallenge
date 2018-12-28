# RobotChallenge

This code allows to perform a sample training on [RGB-D Object Database](https://rgbd-dataset.cs.washington.edu/dataset.html)

## ROD_to_image_folder.py
It's a script to adapt the database format to be loaded by PyTorch ImageFolder Dataset class.

## main.py
It's the script to run the toy example. There are a lot of parameters but the only useful are:
* frozen (if perform feature extraction or finetuning)
* epochs (number of epochs)
* visdom environment (the name of the env)#
* test (you can choose whether to perform a test or to perform also the training)

## training.py
It's the code to perform the training.
The training hyperparameters are:
  * Random resizing crop every image to 224*224 and normalization (with imagenet mean and std)
  * Optimizer is SGD (default are lr=10e-3 and momentum=0.9)
  * Scheduler is set to decrease by 10 the lr after 40 steps
  * loss function is Cross Entropy
  
