#!/usr/bin/env bash
source activate fabio_robot

export CUDA_VISIBLE_DEVICES=2

#split3
python main.py  --prefix ROD_models/S3LR3BS64 --folder /home/fabio/robot_challenge/rod/split3 --epochs 30 --visdom ROD3_LR3BS64 --lr 0.001
python main.py  --prefix ROD_models/S3LR4BS64 --folder /home/fabio/robot_challenge/rod/split3 --epochs 30 --visdom ROD3_LR4BS64 --lr 0.0001
python main.py  --prefix ROD_models/S3LR3BS128 --folder /home/fabio/robot_challenge/rod/split3 --epochs 30 --visdom ROD3_LR3BS128 --lr 0.001 --bs 128
python main.py  --prefix ROD_models/S3LR4BS128 --folder /home/fabio/robot_challenge/rod/split3 --epochs 30 --visdom ROD3_LR4BS128 --lr 0.0001 --bs 128


