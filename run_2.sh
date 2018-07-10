#!/usr/bin/env bash
source activate fabio_robot

export CUDA_VISIBLE_DEVICES=1

#split2
python main.py  --prefix ROD_models/S2LR3BS64 --folder /home/fabio/robot_challenge/rod/split2 --epochs 30 --visdom ROD2_LR3BS64 --lr 0.001
python main.py  --prefix ROD_models/S2LR4BS64 --folder /home/fabio/robot_challenge/rod/split2 --epochs 30 --visdom ROD2_LR4BS64 --lr 0.0001
python main.py  --prefix ROD_models/S2LR3BS128 --folder /home/fabio/robot_challenge/rod/split2 --epochs 30 --visdom ROD2_LR3BS128 --lr 0.001 --bs 128
python main.py  --prefix ROD_models/S2LR4BS128 --folder /home/fabio/robot_challenge/rod/split2 --epochs 30 --visdom ROD2_LR4BS128 --lr 0.0001 --bs 128
