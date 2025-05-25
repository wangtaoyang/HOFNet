import os
from pathlib import Path
import hofnet

# real_hof + hofdiff - fold2
# real_hof - fold1
fold = 'fold0'
devices = [4]
max_epochs = 200
batch_size = 32
seed = 0               # default seeds
BASE_DATA = './data'
BASE_LOG = './logs'
root_dataset = f'{BASE_DATA}/HOF_solvent/{fold}'
task = 'solvent'
downstream = task
log_dir = f'{BASE_LOG}/solvent/fold{fold}/finetune'
os.makedirs(log_dir, exist_ok=True)
cifs_path = './data/hof_database_cifs_raw/total'
load_path = './ckpt/pretrain_real_hofdiff_best.ckpt'  # Pretrained model path

hofnet.run(
    root_dataset, downstream, log_dir=log_dir,                   
    max_epochs=max_epochs, batch_size=batch_size, devices=devices, 
    cifs_path=cifs_path, loss_names="solvent_classification", num_workers=4, load_path=load_path
)