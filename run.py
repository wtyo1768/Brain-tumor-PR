from comet_ml import Experiment
import pandas as pd 
from torch.utils.data import DataLoader
import numpy as np
import argparse
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from cfg import *
from loader import PR_Dataset
from model import cls


experiment = Experiment(
    api_key=COMET_APT_KEY,
    project_name=COMET_PROJECT_NAME,
    workspace=COMET_WORK_SPACE,
)

parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', type=int, default=32)
parser.add_argument('--LR', type=float, default=4e-5)
parser.add_argument('--i', type=int, default=0)
parser.add_argument('--cls', type=str,)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--fold', type=int, default=-1)
parser.add_argument('--MRI', type=str, default='T1')

parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()


df = pd.read_excel(xls_file, sheet_name='Sheet2')
seed = np.random.randint(66) if args.seed==-1 else args.seed
metric = []
K = K_FOLD
for i, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=K, random_state=seed, shuffle=True).split(df, df['Progression/Recurrence (Yes:1 No:0)'])):
    
    if not args.fold==-1:
        if not i==args.fold:
            continue
    print(f'-----------fold{i}-------------')
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_ds = PR_Dataset(train_df, args.MRI, eval_mode=False)
    val_ds = PR_Dataset(val_df, args.MRI, eval_mode=True)

    checkpoint = ModelCheckpoint(
        monitor='val_acc', mode='max', 
        save_top_k=1, filename='{epoch}-{val_acc:.3f}-{f1:.3f}'#, verbose=True
    )
    es = EarlyStopping(
        monitor='val_acc', min_delta=0.00,
        patience=3, verbose=False, mode='max'
    )
    trainer = pl.Trainer.from_argparse_args(
        args,  log_every_n_steps=10, gpus=1,
        #logger=False,progress_bar_refresh_rate=0,
        callbacks=[checkpoint,],        
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.BATCH_SIZE, 
        num_workers=16, 
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.BATCH_SIZE, 
        num_workers=16, 
        shuffle=False,
    )
    model = cls(args, class_weight=train_ds.class_weight, enc=args.cls, p=args.i==0 and i==0, expe=experiment, run=i)
    trainer.fit(
        model, 
        train_loader, 
        val_loader,
    )
    ## Evaluation
    print('path', checkpoint.best_model_path)
    model = cls.load_from_checkpoint(checkpoint.best_model_path, class_weight=train_ds.class_weight, enc=args.cls, expe=experiment)
    pred = trainer.validate(model, val_loader)
    metric.append(pred[0])
    # print(pred[0])


if args.fold==-1:
    for k in metric[0].keys():
        if k=='val_loss':continue
        r = 0
        for e in metric: r += e[k]
        print(f'cv_{k}', ':', r/K)

print('seed:', seed)

