#!/usr/bin/env bash
source activate fabio_robot

export CUDA_VISIBLE_DEVICES=0

#split 1
python main.py  --prefix ROD_models/S1LR23BS64S20E30 --folder /home/fabio/robot_challenge/rod/split1 --step 20 --epochs 30 --visdom ROD1_xRR23BS64S20E30 --lr 0.01 --decay 0.00005
#python main.py  --prefix ROD_models/S1LR4BS64S40E60 --folder /home/fabio/robot_challenge/rod/split1 --step 40 --epochs 60 --visdom ROD1_LR4BS64S40E60 --lr 0.0001
