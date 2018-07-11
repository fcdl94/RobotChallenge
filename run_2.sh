#!/usr/bin/env bash
source activate fabio_robot

export CUDA_VISIBLE_DEVICES=1

#split2
python main.py  --prefix ROD_models/S2LR3BS64S15E30 --folder /home/fabio/robot_challenge/rod/split2 --step 15 --epochs 30 --visdom ROD2_LR3BS64S15E30 --lr 0.001
python main.py  --prefix ROD_models/S2LR4BS64S40E60 --folder /home/fabio/robot_challenge/rod/split2 --step 40 --epochs 60 --visdom ROD2_LR4BS64S40E60 --lr 0.0001