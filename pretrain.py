import hofnet
import os
import hofnet

# real_hof + hofdiff - fold2
# real_hof - fold1
fold = 'fold4'
devices = [5]
max_epochs = 2000
batch_size = 8
root_dataset = f'./data/HOF_pretrain/{fold}'
cifs_path = './data/HOF_cif/cif/total'
task = 'vfp'
downstream = task

log_dir = f'./logs/HOF_pretrain/{fold}'
load_path = None
os.makedirs(log_dir, exist_ok=True)

hofnet.run(root_dataset, downstream, log_dir=log_dir,                   
                max_epochs=max_epochs, batch_size=batch_size, devices=devices, loss_names=['hbond', 'fp', 'vfp'],
                cifs_path=cifs_path, fold=fold, load_path=load_path, learning_rate=1e-5, early_stop_patience=100)