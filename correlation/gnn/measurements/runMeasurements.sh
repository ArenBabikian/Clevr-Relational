#!/usr/bin/env sh

/cygdrive/c/ProgramData/Anaconda3/python --version
python correlation/gnn/train.py --config_fp correlation/gnn/measurements/rgcn-real.yaml
python correlation/gnn/train.py --config_fp correlation/gnn/measurements/rgcn-rand.yaml 
python correlation/gnn/train.py --config_fp correlation/gnn/measurements/gat-real.yaml
python correlation/gnn/train.py --config_fp correlation/gnn/measurements/gat-rand.yaml
# python correlation/gnn/train.py --config_fp correlation/gnn/measurements/rgcn-rand.yaml & python correlation/gnn/train.py --config_fp correlation/gnn/measurements/gat-rand.yaml

# LOGDIR=correlation/gnn/_output/logs/default
# # python -m tensorboard.main --logdir=gat-rand:$LOGDIR/version_<#>
# python -m tensorboard.main --logdir=rgcn-real:correlation/gnn/measurements/_out/logs/default/version_0,rgcn-rand:correlation/gnn/measurements/_out/logs/default/version_1,gat-real:correlation/gnn/measurements/_out/logs/default/version_2,gat-rand:correlation/gnn/measurements/_out/logs/default/version_3