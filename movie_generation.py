import numpy as np
import pandas as pd

import os
import sys
import subprocess
def do(cmd, get=False, show=True):
    if get:
        out = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0].decode()
        if show:
            print(out, end="")
        return out
    else:
        return subprocess.Popen(cmd, shell=True).wait()

file_path = os.path.realpath(__file__)
script_folder = os.path.dirname(file_path)
# print(file_path, script_folder)

import argparse
parser = argparse.ArgumentParser(description="/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/relax/bin/python movie_generation.py results/s21_3/index0_idx_0/ 1+2 --device 3")

parser.add_argument('prediction_result_path', type=str, default='results/test/index0_idx_0', help='informative name used to name result folder')
parser.add_argument('rank', type=str, default="1", help='specify the sample to generate movie.\
             (the samples are sorted by their confidence, with rank 1 being considered the best prediction by the model, rank 40 the worst), \
             could give multiple. for example 1+2+3')
parser.add_argument('--device', type=int, default=0, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--python', type=str, default='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022/bin/python', help='point to the python in dynamicbind env.')
parser.add_argument('--relax_python', type=str, default='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/relax/bin/python', help='point to the python in relax env.')
parser.add_argument('--inference_steps', type=int, default=20, help='num of coordinate updates. (movie frames)')

args = parser.parse_args()

python = args.python
relax_python = args.relax_python
os.environ['PATH'] = os.path.dirname(relax_python) + ":" + os.environ['PATH']

for rank in args.rank.split("+"):
    cmd = f"{python} {script_folder}/save_reverseprocess.py --pklFile {args.prediction_result_path}/rank{rank}_reverseprocess_data_list.pkl"
    do(cmd)

cmd = f"CUDA_VISIBLE_DEVICES={args.device} {relax_python} {script_folder}/relax_vis.py --rank {args.rank} --prediction_result_path {args.prediction_result_path} --inference_steps {args.inference_steps}"
do(cmd)