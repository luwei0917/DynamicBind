import copy
import os
import torch
import shutil

import time
from argparse import ArgumentParser, Namespace, FileType
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
# from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser

from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, AddHs

from datasets.process_mols import read_molecule, generate_conformer, write_mol_with_coords
from utils.visualise import LigandToPDB, modify_pdb, receptor_to_pdb, save_protein
# from utils.relax import openmm_relax
from tqdm import tqdm
import datetime
from contextlib import contextmanager

from multiprocessing import Pool as ThreadPool

import random
import pickle


@contextmanager
def Timer(title):
    'timing function'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"%(title, (datetime.datetime.now() - t0).seconds))
    return None

RDLogger.DisableLog('rdApp.*')
import yaml
parser = ArgumentParser()
parser.add_argument('--results_path', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
parser.add_argument('--samples_per_complex', type=int, default=1, help='Number of samples to generate')
parser.add_argument('--pklFile', type=str, default="", help='specify the pkl files.')

def single_sample_save(pklFile):
    write_dir = os.path.dirname(pklFile)
    fn = os.path.basename(pklFile)
    rank = fn.split('_')[0]
    with open(os.path.join(write_dir,fn),'rb') as f:
        data_step = pickle.load(f)
    lig, receptor_pdb = data_step[0]
    pdb_or_cif = receptor_pdb.get_full_id()[0]
    for idx, data in enumerate(data_step[1:]):
        mol_pred = copy.deepcopy(lig)
        ligandFile = os.path.join(write_dir, f'{rank}_ligand_step{idx+1}.sdf')
        write_mol_with_coords(mol_pred, (data['ligand'].pos + data.original_center).numpy(), ligandFile)
        new_receptor_pdb = copy.deepcopy(receptor_pdb)
        modify_pdb(new_receptor_pdb,data)
        pdbFile = os.path.join(write_dir, f'{rank}_receptor_step{idx+1}.{pdb_or_cif}')
        save_protein(new_receptor_pdb,pdbFile)


def save(write_dir):
    # file_paths = sorted(os.listdir(write_dir))
    # for fn in file_paths:
    for i in range(args.samples_per_complex):
        fn = f'rank{i+1}_reverseprocess_data_list.pkl'
        single_sample_save(os.path.join(write_dir,fn))


args = parser.parse_args()
if args.pklFile != "":
    single_sample_save(args.pklFile)
else:
    results_path_containments = os.listdir(args.results_path)
    results_path_containments = [x for x in results_path_containments if x != 'affinity_prediction.csv']

    for rp in tqdm(results_path_containments):
        if not rp.startswith('index'):
            continue
        write_dir = os.path.join(args.results_path, rp)
        save(write_dir)
