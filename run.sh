#!/usr/bin/env bash
source activate MultiDomainLearning

python main.py  --prefix ROD_models/try02 --epochs 10 --visdom ROD_LR2 --lr 0.01
python main.py  --prefix ROD_models/try03 --epochs 10 --visdom ROD_LR3 --lr 0.001
python main.py  --prefix ROD_models/try04 --epochs 10 --visdom ROD_LR4 --lr 0.0001
