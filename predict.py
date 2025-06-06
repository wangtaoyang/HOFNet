from pathlib import Path
import pandas as pd
import numpy as np
import ast
from hofnet import predict  # Assuming hofnet is properly installed and configured


# === Configuration ===
fold = 'fold0'
task = 'solvent'
version = 0
seed = 0
checkpoint = 'best'
log_dir = Path('./logs') / task / fold
load_path = Path('/mnt/user2/wty/HOF/logs/solvent/fold0/nh_na/pretrained_mof_seed0_from_best/version_44/checkpoints/best.ckpt')
# load_path = Path('/mnt/user2/wty/HOFNet/logs/solvent/foldfold0/finetune/pretrained_mof_seed0_from_best/version_0/checkpoints/best.ckpt')
root_dataset = f'./data/HOF_solvent/{fold}'
# cifs_path = ".data/hof_database_cifs_raw/total"
cifs_path = "/mnt/user2/wty/HOF/MOFDiff/mofdiff/data/mof_models/mof_models/bwdb_hoff/hofchecker/total"
devices = [0]
losses_name = 'solvent_classification'

# === Prediction ===
predict(root_dataset, load_path, cifs_path=cifs_path, downstream=task,
        save_dir=log_dir, devices=devices, loss_names=losses_name)

# === Post-process ===
solvent_csv = './solvent.csv'
result_csv = log_dir / 'updated_classification_results.csv'
df = pd.read_csv(log_dir / 'test_prediction.csv')

df['solvent_classification_logits'] = df['solvent_classification_logits'].apply(ast.literal_eval)
df['solvent_classification_labels'] = df['solvent_classification_labels'].apply(ast.literal_eval)
df.to_csv(result_csv, index=False)

solvents_df = pd.read_csv(solvent_csv)
solvents_df['properties'] = solvents_df[['LogP', 'Area', 'donors', 'acceptors', 'point']].values.tolist()

def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def multi_top_accuracy(true_labels, predictions):
    correct = sum(true in preds for true, preds in zip(true_labels, predictions))
    return correct / len(true_labels)

for col in ['top1_solvent', 'top2_solvent', 'top3_solvent', 'true_solvent']:
    df[col] = ""

for index, row in df.iterrows():
    pred_vec = row['solvent_classification_logits']
    true_vec = row['solvent_classification_labels']
    mse_scores = solvents_df['properties'].apply(lambda x: mse(x, pred_vec))
    top3_indices = mse_scores.nsmallest(3).index
    top3_labels = solvents_df.loc[top3_indices, 'solvent1_label'].tolist()

    df.at[index, 'top1_solvent'] = top3_labels[0]
    df.at[index, 'top2_solvent'] = top3_labels[1]
    df.at[index, 'top3_solvent'] = top3_labels[2]

    true_mse_scores = solvents_df['properties'].apply(lambda x: mse(x, true_vec))
    best_match_index = true_mse_scores.idxmin()
    df.at[index, 'true_solvent'] = solvents_df.at[best_match_index, 'solvent1_label']

df.to_csv(result_csv, index=False)

true_labels = df['true_solvent'].tolist()
top1_preds = df['top1_solvent'].tolist()
top2_preds = df[['top1_solvent', 'top2_solvent']].values.tolist()
top3_preds = df[['top1_solvent', 'top2_solvent', 'top3_solvent']].values.tolist()

top1_accuracy = multi_top_accuracy(true_labels, [[pred] for pred in top1_preds])
top2_accuracy = multi_top_accuracy(true_labels, top2_preds)
top3_accuracy = multi_top_accuracy(true_labels, top3_preds)

print(f"Top1 Accuracy: {top1_accuracy:.4f}")
print(f"Top2 Accuracy: {top2_accuracy:.4f}")
print(f"Top3 Accuracy: {top3_accuracy:.4f}")
