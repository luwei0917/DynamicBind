import os
from argparse import FileType, ArgumentParser
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/PDBBind_af2_aligned.csv', help='')
parser.add_argument('--chain_cutoff', type=int, default=10, help='')
parser.add_argument('--out_file', type=str, default="data/pdbbind_sequences.fasta")
args = parser.parse_args()

cutoff = args.chain_cutoff
data_path = args.data_path
df = pd.read_csv(data_path)
#%%
from Bio import SeqIO
biopython_pdbparser = PDBParser()
biopython_cifparser = MMCIFParser()

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

sequences = []
ids = []
for i in tqdm(df.index):
    name = df.loc[i,'name']
    rec_path = df.loc[i,'crystal_protein_path']
    if rec_path[-4:] == '.pdb':
        structure = biopython_pdbparser.get_structure('pdb', rec_path)
    elif rec_path[-4:] == '.cif':
        structure = biopython_cifparser.get_structure('cif', rec_path)
    else:
        raise "protein is not pdb or cif"
    structure = structure[0]
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex ', name, '. Replacing it with a dash - .')
        sequences.append(seq)
        ids.append(f'{name}_chain_{i}')
records = []
for (index, seq) in zip(ids,sequences):
    record = SeqRecord(Seq(seq), str(index))
    record.description = ''
    records.append(record)
SeqIO.write(records, args.out_file, "fasta")
