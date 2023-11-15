from argparse import ArgumentParser, Namespace, FileType
from Bio.PDB import PDBParser,MMCIFParser
from Bio.PDB import PDBIO, MMCIFIO, Select
import os,copy
from tqdm import tqdm
from utils.relax import openmm_relax, openmm_relax_protein_only

from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
import rdkit.Chem
from rdkit import Geometry
import numpy as np
from collections import defaultdict


class LigandToPDB:
    def __init__(self, mol):
        self.parts = defaultdict(dict)
        self.mol = copy.deepcopy(mol)
        [self.mol.RemoveConformer(j) for j in range(mol.GetNumConformers()) if j]
    def add(self, coords, order, part=0, repeat=1):
        if type(coords) in [rdkit.Chem.Mol, rdkit.Chem.RWMol]:
            block = MolToPDBBlock(coords).split('\n')[:-2]
            self.parts[part][order] = {'block': block, 'repeat': repeat}
            return
        elif type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.double().numpy()
        for i in range(coords.shape[0]):
            self.mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
        block = MolToPDBBlock(self.mol).split('\n')[:-2]
        self.parts[part][order] = {'block': block, 'repeat': repeat}

    def write(self, path=None, limit_parts=None):
        is_first = True
        str_ = ''
        for part in sorted(self.parts.keys()):
            if limit_parts and part >= limit_parts:
                break
            part = self.parts[part]
            keys_positive = sorted(filter(lambda x: x >=0, part.keys()))
            keys_negative = sorted(filter(lambda x: x < 0, part.keys()))
            keys = list(keys_positive) + list(keys_negative)
            for key in keys:
                block = part[key]['block']
                times = part[key]['repeat']
                for _ in range(times):
                    if not is_first:
                        block = [line for line in block if 'CONECT' not in line]
                    is_first = False
                    str_ += f'MODEL {key}\n'
                    str_ += '\n'.join(block)
                    str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)

def save_protein(s, proteinFile, ca_only=False):
    if proteinFile[-3:] == 'pdb':
        io = PDBIO()
    elif proteinFile[-3:] == 'cif':
        io = MMCIFIO()
    else:
        raise 'protein is not pdb or cif'
    class MySelect(Select):
        def accept_atom(self, atom):
            if atom.get_name() == 'CA':
                return True
            else:
                return False
    class RemoveHs(Select):
        def accept_atom(self, atom):
            if atom.element != 'H':
                return True
            else:
                return False
    io.set_structure(s)
    if ca_only:
        io.save(proteinFile, MySelect())
    else:
        io.save(proteinFile, RemoveHs())
    return None


def single_sample_movie(file_names, write_dir, rank, inference_steps, ):
    for step in range(inference_steps+1):
        ligandFile = os.path.join(write_dir, [fn for fn in file_names if f'rank{rank}_ligand_step{step+1}.' in fn][0])
        pdbFile = os.path.join(write_dir, [fn for fn in file_names if f'rank{rank}_receptor_step{step+1}.' in fn][0])
        if step == 0:
            lig = rdkit.Chem.SDMolSupplier(ligandFile, sanitize=False, removeHs=False)[0]
            relaxed_ligand = LigandToPDB(lig)
            lig_pos = lig.GetConformer().GetPositions()
            relaxed_ligand.add(lig_pos, 1, 0)

            pdb_or_cif = pdbFile[-3:]
            if pdb_or_cif == 'pdb':
                parser = PDBParser()
            elif pdb_or_cif == 'cif':
                parser = MMCIFParser()
            else:
                raise 'protein is not pdb or cif'
            relaxed_protein = parser.get_structure(pdb_or_cif, pdbFile)
            continue
        fixed_pdbFile = os.path.join(write_dir, f'fixed.{pdb_or_cif}')
        relaxed_proteinFile = os.path.join(write_dir, f'rank{rank}_receptor_step{step+1}_relaxed.{pdb_or_cif}')
        gap_mask = "none"
        stiffness = 1000
        ligand_stiffness = 3000
        relaxed_complexFile = "none"
        use_gpu = True #if torch.cuda.is_available() else False
        if step < inference_steps-1:
            relaxed_ligandFile = ligandFile
            try:
                retry = 0
                ret = openmm_relax_protein_only((pdbFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, use_gpu))
                while ret['efinal'] > 0 and retry < 5:
                    # if ret['einit'] > 0 and ret['efinal'] / ret['einit'] < 0.1:
                    #     break
                    ret = openmm_relax_protein_only((relaxed_proteinFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, use_gpu))

                    retry += 1
                # print(ret['einit'],ret['efinal'])
            except Exception as e:
                print(e, "relax fail, use original instead")
                _ = os.system(f"cp {pdbFile} {relaxed_proteinFile}")

        else:
            relaxed_ligandFile = os.path.join(write_dir, [fn for fn in file_names if f'rank{rank}_ligand_lddt' in fn and 'relaxed' in fn][0])
            relaxed_proteinFile = os.path.join(write_dir, [fn for fn in file_names if f'rank{rank}_receptor_lddt' in fn and 'relaxed' in fn][0])
            # relaxed_ligandFile = os.path.join(write_dir, f'rank{rank}_ligand_step{step+1}_relaxed.sdf')
            # try:
            #     retry = 0
            #     ret = openmm_relax((pdbFile, ligandFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu))
            #     while ret['efinal'] > 0 and retry < 5:
            #         # print(ret['einit'],ret['efinal'])
            #         # if ret['einit'] > 0 and ret['efinal'] / ret['einit'] < 0.001:
            #         #     break
            #         ret = openmm_relax((relaxed_proteinFile, ligandFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu))
            #         retry += 1
            #     # print(ret['einit'],ret['efinal'])
            #     # if ret['efinal'] > 0:
            #     #     ret = openmm_relax_protein_only((pdbFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, use_gpu))
            #     #     print(ret['einit'],ret['efinal'])
            #     #     ret = openmm_relax((relaxed_proteinFile, ligandFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu))
            #     #     print(ret['einit'],ret['efinal'])
            # except Exception as e:
            #     print(e, "relax fail, use original instead")
            #     _ = os.system(f"cp {pdbFile} {relaxed_proteinFile}")
            #     _ = os.system(f"cp {ligandFile} {relaxed_ligandFile}")

        lig = rdkit.Chem.SDMolSupplier(relaxed_ligandFile, sanitize=False, removeHs=False)[0]
        lig_pos = lig.GetConformer().GetPositions()
        relaxed_ligand.add(lig_pos, 1, step)

        s = parser.get_structure(pdb_or_cif, relaxed_proteinFile)[0]
        s.id = step
        s.serial_num = step + 1
        relaxed_protein.add(s)
    vis_ligandFile = os.path.join(write_dir, f'rank{rank}_ligand_reverseprocess_relaxed.pdb')
    relaxed_ligand.write(vis_ligandFile)
    save_protein(relaxed_protein,os.path.join(write_dir, f'rank{rank}_receptor_reverseprocess_relaxed.{pdb_or_cif}'))

parser = ArgumentParser()



parser.add_argument('--rank', type=str, default="", help='specify the sample to generate movie.\
             (the samples are sorted by their confidence, with rank 1 being considered the best prediction by the model, rank 40 the worst), \
             could give multiple. for example 1+2+3')
parser.add_argument('--prediction_result_path', type=str, default='results/test/index0_idx_0', help='informative name used to name result folder')

parser.add_argument('--inference_steps', type=int, default=20, help='num of coordinate updates. (movie frames)')
parser.add_argument('--results_path', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
parser.add_argument('--samples_per_complex', type=int, default=1, help='Number of samples to generate')

args = parser.parse_args()

if args.rank != "":
    write_dir = args.prediction_result_path
    file_names = os.listdir(write_dir)
    for rank in args.rank.split("+"):
        single_sample_movie(file_names, write_dir, rank, args.inference_steps)
else:
    results_path_containments = os.listdir(args.results_path)
    results_path_containments = [x for x in results_path_containments if x != 'affinity_prediction.csv']

    def relax(write_dir):
        file_names = os.listdir(write_dir)
        for rank in range(args.samples_per_complex):
            single_sample_movie(file_names, write_dir, rank+1, args.inference_steps)
    for rp in tqdm(results_path_containments):
        if not rp.startswith('index'):
            continue
        write_dir = os.path.join(args.results_path, rp)
        relax(write_dir)