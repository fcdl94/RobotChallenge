# RobotChallenge

This code has been released for the RGBD Triathlon benchmark.

## ROD_to_image_folder.py
It's a script to adapt the database format to be loaded by PyTorch ImageFolder Dataset class.

## main.py
It's the script to run the toy example. You can start from here to add your method or evaluate methods.

## training.py
It's the code to perform the training.
The training hyperparameters are:
  * Random resizing crop every image to 224*224 and normalization (with imagenet mean and std)
  * Optimizer is SGD (default are lr=10e-3 and momentum=0.9)
  * Scheduler is set to decrease by 10 the lr after 40 steps
  * loss function is Cross Entropy
  
