import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from argparse import ArgumentParser, Namespace, FileType
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--out_dir', type=str, default='./data/', help='gpu id')

args = parser.parse_args()

all_df = []
for path in tqdm(os.listdir(args.out_dir)):
    if os.path.exists(f'{args.out_dir}/{path}/affinity_prediction.csv'):
        df = pd.read_csv(f'{args.out_dir}/{path}.csv')
        pred = pd.read_csv(f'{args.out_dir}/{path}/affinity_prediction.csv')
        pred = pred.set_index(pred['name'].apply(lambda x:int(x.split('_')[1]))).rename(columns={'affinity':'affinity_pred'})
        df = df.merge(pred,left_index=True,right_index=True)
        all_df.append(df)
all_df = pd.concat(all_df)
all_df.to_csv(f'{args.out_dir}/affinity_prediction.csv',index=False)
