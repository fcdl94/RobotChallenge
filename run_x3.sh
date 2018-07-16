#!/usr/bin/env bash
source activate fabio_robot

export CUDA_VISIBLE_DEVICES=2

#python main.py  --prefix RODx_models/S1LR23BS64t1 --folder /home/fabio/robot_challenge/rod/split1 --visdom RODx_S1LR23BS32T1 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32 

#python main.py  --prefix RODx_models/S1LR23BS64t2 --folder /home/fabio/robot_challenge/rod/split1 --visdom RODx_S1LR23BS32T2 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32 

#python main.py  --prefix RODx_models/S1LR23BS64t3 --folder /home/fabio/robot_challenge/rod/split1 --visdom RODx_S1LR23BS32T3 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32 

#python main.py  --prefix RODx_models/S2LR23BS64t1 --folder /home/fabio/robot_challenge/rod/split2 --visdom RODx_S2LR23BS32T1 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32
#python main.py  --prefix RODx_models/S2LR23BS64t2 --folder /home/fabio/robot_challenge/rod/split2 --visdom RODx_S2LR23BS32T2 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32
#python main.py  --prefix RODx_models/S2LR23BS64t3 --folder /home/fabio/robot_challenge/rod/split2 --visdom RODx_S2LR23BS32T3 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32


python main.py  --prefix RODx_models/S3LR23BS64t1 --folder /home/fabio/robot_challenge/rod/split3 --visdom RODx_S3LR23BS32T1 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32
python main.py  --prefix RODx_models/S3LR23BS64t2 --folder /home/fabio/robot_challenge/rod/split3 --visdom RODx_S3LR23BS32T2 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32
python main.py  --prefix RODx_models/S3LR23BS64t3 --folder /home/fabio/robot_challenge/rod/split3 --visdom RODx_S3LR23BS32T3 --step 20 --epochs 30 --lr 0.005 --decay 0.00005 --bs 32
