import hofnet
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import hofnet
from hofnet.examples import example_path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import json

# data root and downstream from example
fold = 'fold5'
devices = [0,1,2,3]
max_epochs = 2000
batch_size = 8
root_dataset = f'/mnt/user2/wty/HOF/moftransformer/data/HOF_pretrain_new/{fold}'
cifs_path = '/mnt/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/hofchecker_1/total'
fp_file_path = '/mnt/user2/wty/HOF/moftransformer/data/HOF_pretrain_new/all_fp.json'
task = 'vfp' 
downstream = task

# downstream = None

log_dir = f'/mnt/user2/wty/HOF/logs/HOF_pretrain/{fold}'
load_path = None
# load_path = 'pmtransformer'
os.makedirs(log_dir, exist_ok=True)


hofnet.run(root_dataset, downstream, log_dir=log_dir,                   
                max_epochs=max_epochs, batch_size=batch_size, devices=devices, loss_names=['hbond', 'fp', 'vfp'],
                cifs_path=cifs_path, fold=fold, fp_file_path=fp_file_path,
                load_path=load_path, learning_rate=1e-5)    