from argparse import ArgumentParser, Namespace, FileType
from Bio.PDB import PDBParser,PDBIO
import torch,os,copy
import pandas as pd
from utils.visualise import save_protein
parser = ArgumentParser()

parser.add_argument('--results_path', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

args = parser.parse_args()

results_path_containments = os.listdir(args.results_path)

df = pd.read_csv('./data/test_af2.csv')
for rp in results_path_containments:
    if not rp.startswith('index'):
        continue
    idx = int(rp.split('_')[0][5:])
    # if idx not in [268, 157, 123, 103,  54, 121, 165]:
    #     continue
    print(idx)
    gap_mask = df.loc[idx,'gap_mask']
    write_dir = os.path.join(args.results_path, rp)
    file_paths = sorted(os.listdir(write_dir))
    for rank in range(args.samples_per_complex):
        try:
            protein_file_name = [path for path in file_paths if f'rank{rank+1}_receptor_lddt' in path and 'relaxed.pdb' in path][0]
        except:
            continue
        pdbFile = os.path.join(write_dir, protein_file_name)
        save_pdbFile = os.path.join(write_dir, protein_file_name.replace('relaxed.pdb','relaxed_remove_gap.pdb'))
        s = PDBParser(QUIET=True).get_structure(pdbFile,pdbFile)[0]
        start = 0
        for c in s:
            for i, res in enumerate(list(c.get_residues())):
                if gap_mask[start+i] == '1':
                    c.detach_child(res.id)
            start += (i+1)
        save_protein(s, save_pdbFile)

        try:
            protein_file_name = [path for path in file_paths if f'rank{rank+1}_receptor_reverseprocess_relaxed' in path][0]
        except:
            continue
        pdbFile = os.path.join(write_dir, protein_file_name)
        save_pdbFile = os.path.join(write_dir, protein_file_name.replace('relaxed.pdb','relaxed_remove_gap.pdb'))
        all_s = PDBParser(QUIET=True).get_structure(pdbFile,pdbFile)
        for s in all_s:
            start = 0
            for c in s:
                for i, res in enumerate(list(c.get_residues())):
                    if gap_mask[start+i] == '1':
                        c.detach_child(res.id)
                start += (i+1)
        save_protein(all_s, save_pdbFile)

    protein_file_name = [path for path in file_paths if f'_aligned_to_' in path][0]
    pdbFile = os.path.join(write_dir, protein_file_name)
    save_pdbFile = os.path.join(write_dir, protein_file_name.replace('.pdb','_remove_gap.pdb'))
    s = PDBParser(QUIET=True).get_structure(pdbFile,pdbFile)[0]
    start = 0
    for c in s:
        for i, res in enumerate(list(c.get_residues())):
            if gap_mask[start+i] == '1':
                c.detach_child(res.id)
        start += (i+1)
    save_protein(s, save_pdbFile)

    protein_file_name = [path for path in file_paths if f'af2_' in path][0]
    pdbFile = os.path.join(write_dir, protein_file_name)
    save_pdbFile = os.path.join(write_dir, protein_file_name.replace('.pdb','_remove_gap.pdb'))
    s = PDBParser(QUIET=True).get_structure(pdbFile,pdbFile)[0]
    start = 0
    for c in s:
        for i, res in enumerate(list(c.get_residues())):
            if gap_mask[start+i] == '1':
                c.detach_child(res.id)
        start += (i+1)
    save_protein(s, save_pdbFile)
