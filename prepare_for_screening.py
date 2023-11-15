import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser, Namespace, FileType
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--python', type=str, default='', help='python path')
parser.add_argument('--relax_python', type=str, default='', help='python path')
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path and --ligand parameters')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu id')
parser.add_argument('--batch_size', type=int, default=5000, help='inference batch size')
parser.add_argument('--out_dir', type=str, default='./data/', help='gpu id')
parser.add_argument('--model', type=int, default=1, help='default model version')

args = parser.parse_args()

os.makedirs(args.out_dir,exist_ok=True)
df = pd.read_csv(args.protein_ligand_csv)
batch_size = args.batch_size
gpus = args.gpu_ids.split(',')
gpu_count = len(gpus)
shells = []

python=args.python

for i in range(gpu_count):
    shells.append(open(f'{args.out_dir}/run_screening_job{i}.sh','w'))
    # shells[-1].write('cd ../\n')

for i in range(int(np.ceil(len(df)/batch_size))):
    df[i*batch_size:(i+1)*batch_size].to_csv(f'{args.out_dir}/batch{i}.csv',index=False)
    gpu_idx = i%gpu_count
    shells[gpu_idx].write(f"export MKL_THREADING_LAYER=GNU\n")
    shells[gpu_idx].write(f"{python} run_single_protein_inference_ft.py mask {args.out_dir}/batch{i}.csv -l --model {args.model} --job screening --no_relax --samples_per_complex 3 --savings_per_complex 0 --num_workers 8 --inference_steps 20 --header {args.out_dir}/batch{i} --device {gpus[gpu_idx]} --python {python} --relax_python {args.relax_python}\n")
