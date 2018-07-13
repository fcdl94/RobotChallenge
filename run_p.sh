#!/usr/bin/env bash
source activate fabio_robot

export CUDA_VISIBLE_DEVICES=0

python main.py  --prefix ROD_models/S1LR34BS32NN --folder /home/fabio/robot_challenge/rod/split1 --step 20 --epochs 30 --visdom ROD1_LR5-23BS32 --lr 0.005 --decay 0.00005 --bs 32
# python main.py  --prefix ROD_models/S1LR23BS32NN --folder /home/fabio/robot_challenge/rod/split1 --step 20 --epochs 30 --visdom ROD1_LR1-23BS32 --lr 0.01  --decay 0.00005 --bs 32
# python main.py  --prefix ROD_models/S1LR23BS64D4 --folder /home/fabio/robot_challenge/rod/split1 --step 20 --epochs 30 --visdom ROD1_LR23BS64D5-4 --lr 0.01  --decay 0.0005
