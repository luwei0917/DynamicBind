import numpy as np
import pandas as pd

import os
import sys

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import PDBIO, Select, MMCIFIO
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description="python clean_pdb.py a.pdb a_cleaned.pdb")

parser.add_argument('input', type=str, default='test.pdb', help="protein file that requries cleaning. could also be a csv file contains a column named 'protein_path' ")
parser.add_argument('output', type=str, default='test_cleaned.pdb', help='name of the cleaned file.')
parser.add_argument('--keep_chain', type=str, default='keep_all', help='keep a certain chain or keep all')
parser.add_argument('--no_fix', action='store_true', default=False, help='by default, the protein will be fixed using openMM.')
args = parser.parse_args()

if not args.no_fix:
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile, PDBxFile
        from openmm import app as openmm_app
    except:
        print("should be run with relax python environment")

def remove_hydrogen_pdb(pdbFile, toFile):

    parser = MMCIFParser(QUIET=True) if pdbFile[-4:] == ".cif" else PDBParser(QUIET=True)
    s = parser.get_structure("x", pdbFile)
    class NoHydrogen(Select):
        def accept_atom(self, atom):
            if atom.element == 'H' or atom.element == 'D':
                return False
            return True

    io = MMCIFIO() if toFile[-4:] == ".cif" else PDBIO()
    io.set_structure(s)
    io.save(toFile, select=NoHydrogen())


def save_clean_protein(s, toFile, keep_chain='A', keep_all_protein_chains=True):
    class MySelect(Select):
        def accept_residue(self, residue, keep_chain=keep_chain):
            pdb, _, chain, (hetero, resid, insertion) = residue.full_id
            if keep_all_protein_chains or (chain == keep_chain):
                # if hetero == ' ' or hetero == 'H_MSE':
                if hetero == 'H_MSE':
                    residue.id = (' ', resid, insertion)
                    return True
                elif hetero == ' ':
                    return True
                else:
                    return False
            else:
                return False
        def accept_atom(self, atom):
            # remove altloc atoms.
            return not atom.is_disordered() or atom.get_altloc() == "A"
    if toFile[-4:] == ".cif":
        io = MMCIFIO()
    elif toFile[-4:] == ".pdb":
        io = PDBIO()
    else:
        print("toFile should end with .cif or .pdb")
    io.set_structure(s)
    io.save(toFile, MySelect())
    

def clean_one_pdb(proteinFile, toFile):
    if proteinFile[-4:] == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    s = parser.get_structure(proteinFile, proteinFile)
    if args.keep_chain == 'keep_all':
        save_clean_protein(s, toFile, keep_all_protein_chains=True)
    else:
        save_clean_protein(s, toFile, keep_chain=args.keep_chain, keep_all_protein_chains=False)

    if not args.no_fix:
        pdbFile = toFile
        fixed_pdbFile = toFile
        remove_hydrogen_pdb(pdbFile, fixed_pdbFile)
        fixer = PDBFixer(filename=fixed_pdbFile)
        fixer.removeHeterogens()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        fixer.addMissingHydrogens()
        if pdbFile[-3:] == 'pdb':
            PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdbFile, 'w'),
                            keepIds=True)
        elif pdbFile[-3:] == 'cif':
            PDBxFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdbFile, 'w'),
                            keepIds=True)
        else:
            raise 'protein is not pdb or cif'
        remove_hydrogen_pdb(fixed_pdbFile, fixed_pdbFile)


if args.input[-4:] == ".cif" or args.input[-4:] == ".pdb":
    proteinFile = args.input
    toFile = args.output
    clean_one_pdb(proteinFile, toFile)
elif args.input[-4:] == ".csv":
    input_name = os.path.basename(args.input)[:-4]
    output_name = os.path.basename(args.output)[:-4]
    df = pd.read_csv(args.input)
    if os.path.exists(f'./data/tmp_{output_name}/'):
        # prevent the case when file content changed but file name is unchanged.
        os.system(f"rm -r ./data/tmp_{output_name}/")
    os.makedirs(f'./data/tmp_{output_name}/',exist_ok=False)
    new_protein_path = []
    for i in tqdm(df.index):
        proteinFile = df.loc[i,'protein_path']
        # toFile = f'./data/tmp/{os.path.basename(proteinFile)}'
        toFile = f'./data/tmp_{output_name}/{os.path.basename(proteinFile)}'
        new_protein_path.append(toFile)
        if os.path.exists(toFile):
            continue
        clean_one_pdb(proteinFile, toFile)
        # new_protein_path.append(toFile)
    df['protein_path'] = new_protein_path
    # f'./data/tmp/{os.path.basename(args.protein_ligand_csv)}'
    df.to_csv(args.output,index=False)
else:
    print("unknown input.")
