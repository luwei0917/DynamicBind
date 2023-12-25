from argparse import ArgumentParser, Namespace, FileType
from Bio.PDB import PDBParser,PDBIO
import os,copy
import numpy as np

from multiprocessing import Pool
import tqdm
import time
from tqdm.contrib.concurrent import process_map  # or thread_map
from scipy.spatial.distance import cdist

from rdkit.Chem import AllChem
import rdkit.Chem as Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import Superimposer
from Bio.PDB import PDBIO, Select,NeighborSearch
from openmm import app as openmm_app

from utils.relax import openmm_relax

parser = ArgumentParser()

parser.add_argument('--results_path', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--num_workers', type=int, default=20, help='Number of workers for creating the dataset')
parser.add_argument('--samples_per_complex', type=int, default=1, help='Number of samples to generate')

args = parser.parse_args()

from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
import rdkit.Chem
from rdkit import Geometry
from collections import defaultdict
import copy
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation as R
from Bio.PDB import PDBIO, MMCIFIO, Select

chi_atoms = dict(
    chi1=dict(
        ARG=['N', 'CA', 'CB', 'CG'],
        ASN=['N', 'CA', 'CB', 'CG'],
        ASP=['N', 'CA', 'CB', 'CG'],
        CYS=['N', 'CA', 'CB', 'SG'],
        GLN=['N', 'CA', 'CB', 'CG'],
        GLU=['N', 'CA', 'CB', 'CG'],
        HIS=['N', 'CA', 'CB', 'CG'],
        ILE=['N', 'CA', 'CB', 'CG1'],
        LEU=['N', 'CA', 'CB', 'CG'],
        LYS=['N', 'CA', 'CB', 'CG'],
        MET=['N', 'CA', 'CB', 'CG'],
        PHE=['N', 'CA', 'CB', 'CG'],
        PRO=['N', 'CA', 'CB', 'CG'],
        SER=['N', 'CA', 'CB', 'OG'],
        THR=['N', 'CA', 'CB', 'OG1'],
        TRP=['N', 'CA', 'CB', 'CG'],
        TYR=['N', 'CA', 'CB', 'CG'],
        VAL=['N', 'CA', 'CB', 'CG1'],
    ),
    altchi1=dict(
        VAL=['N', 'CA', 'CB', 'CG2'],
    ),
    chi2=dict(
        ARG=['CA', 'CB', 'CG', 'CD'],
        ASN=['CA', 'CB', 'CG', 'OD1'],
        ASP=['CA', 'CB', 'CG', 'OD1'],
        GLN=['CA', 'CB', 'CG', 'CD'],
        GLU=['CA', 'CB', 'CG', 'CD'],
        HIS=['CA', 'CB', 'CG', 'ND1'],
        ILE=['CA', 'CB', 'CG1', 'CD1'],
        LEU=['CA', 'CB', 'CG', 'CD1'],
        LYS=['CA', 'CB', 'CG', 'CD'],
        MET=['CA', 'CB', 'CG', 'SD'],
        PHE=['CA', 'CB', 'CG', 'CD1'],
        PRO=['CA', 'CB', 'CG', 'CD'],
        TRP=['CA', 'CB', 'CG', 'CD1'],
        TYR=['CA', 'CB', 'CG', 'CD1'],
    ),
    altchi2=dict(
        ASP=['CA', 'CB', 'CG', 'OD2'],
        LEU=['CA', 'CB', 'CG', 'CD2'],
        PHE=['CA', 'CB', 'CG', 'CD2'],
        TYR=['CA', 'CB', 'CG', 'CD2'],
    ),
    chi3=dict(
        ARG=['CB', 'CG', 'CD', 'NE'],
        GLN=['CB', 'CG', 'CD', 'OE1'],
        GLU=['CB', 'CG', 'CD', 'OE1'],
        LYS=['CB', 'CG', 'CD', 'CE'],
        MET=['CB', 'CG', 'SD', 'CE'],
    ),
    chi4=dict(
        ARG=['CG', 'CD', 'NE', 'CZ'],
        LYS=['CG', 'CD', 'CE', 'NZ'],
    ),
    chi5=dict(
        ARG=['CD', 'NE', 'CZ', 'NH1'],
    ),
)
chi_names = ['chi%s'%i for i in range(1,6)]
atom_order_dict = {
 'GLN': 'N,CA,C,CB,O,CG,CD,NE2,OE1',
 'PRO': 'N,CA,C,CB,O,CG,CD',
 'THR': 'N,CA,C,CB,O,CG2,OG1',
 'LEU': 'N,CA,C,CB,O,CG,CD1,CD2',
 'SER': 'N,CA,C,CB,O,OG',
 'HIS': 'N,CA,C,CB,O,CG,CD2,ND1,CE1,NE2',
 'ASN': 'N,CA,C,CB,O,CG,ND2,OD1',
 'MET': 'N,CA,C,CB,O,CG,SD,CE',
 'ILE': 'N,CA,C,CB,O,CG1,CG2,CD1',
 'ALA': 'N,CA,C,CB,O',
 'GLY': 'N,CA,C,O',
 'VAL': 'N,CA,C,CB,O,CG1,CG2',
 'GLU': 'N,CA,C,CB,O,CG,CD,OE1,OE2',
 'LYS': 'N,CA,C,CB,O,CG,CD,CE,NZ',
 'CYS': 'N,CA,C,CB,O,SG',
 'TRP': 'N,CA,C,CB,O,CG,CD1,CD2,CE2,CE3,NE1,CH2,CZ2,CZ3',
 'ARG': 'N,CA,C,CB,O,CG,CD,NE,NH1,NH2,CZ',
 'TYR': 'N,CA,C,CB,O,CG,CD1,CD2,CE1,CE2,OH,CZ',
 'PHE': 'N,CA,C,CB,O,CG,CD1,CD2,CE1,CE2,CZ',
 'ASP': 'N,CA,C,CB,O,CG,OD1,OD2'}

# a tuple, left bond atom, right bond atom, list of atoms should rotate with the bond.
chi1_bond_dict = {
    "ALA":None,
    "ARG":("CA", "CB", ["CG", "CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN":("CA", "CB", ["CG", "ND2", "OD1"]),
    "ASP":("CA", "CB", ["CG", "OD1", "OD2"]),
    "CYS":("CA", "CB", ["SG"]),
    "GLN":("CA", "CB", ["CG", "CD", "NE2", "OE1"]),
    "GLU":("CA", "CB", ["CG", "CD", "OE1", "OE2"]),
    "GLY":None,
    "HIS":("CA", "CB", ["CG", "CD2", "ND1", "CE1", "NE2"]),
    "ILE":("CA", "CB", ["CG1", "CG2", "CD1"]),
    "LEU":("CA", "CB", ["CG", "CD1", "CD2"]),
    "LYS":("CA", "CB", ["CG", "CD", "CE", "NZ"]),
    "MET":("CA", "CB", ["CG", "SD", "CE"]),
    "PHE":("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO":("CA", "CB", ["CG", "CD"]),
    "SER":("CA", "CB", ["OG"]),
    "THR":("CA", "CB", ["CG2", "OG1"]),
    "TRP":("CA", "CB", ["CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR":("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL":("CA", "CB", ["CG1", "CG2"])
}

chi2_bond_dict = {
    "ALA":None,
    "ARG":("CB", "CG", ["CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN":("CB", "CG", ["ND2", "OD1"]),
    "ASP":("CB", "CG", ["OD1", "OD2"]),
    "CYS":None,
    "GLN":("CB", "CG", ["CD", "NE2", "OE1"]),
    "GLU":("CB", "CG", ["CD", "OE1", "OE2"]),
    "GLY":None,
    "HIS":("CB", "CG", ["CD2", "ND1", "CE1", "NE2"]),
    "ILE":("CB", "CG1", ["CD1"]),
    "LEU":("CB", "CG", ["CD1", "CD2"]),
    "LYS":("CB", "CG", ["CD", "CE", "NZ"]),
    "MET":("CB", "CG", ["SD", "CE"]),
    "PHE":("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO":("CB", "CG", ["CD"]),
    "SER":None,
    "THR":None,
    "TRP":("CB", "CG", ["CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR":("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL":None,
}


chi3_bond_dict = {
    "ALA":None,
    "ARG":("CG", "CD", ["NE", "NH1", "NH2", "CZ"]),
    "ASN":None,
    "ASP":None,
    "CYS":None,
    "GLN":("CG", "CD", ["NE2", "OE1"]),
    "GLU":("CG", "CD", ["OE1", "OE2"]),
    "GLY":None,
    "HIS":None,
    "ILE":None,
    "LEU":None,
    "LYS":("CG", "CD", ["CE", "NZ"]),
    "MET":("CG", "SD", ["CE"]),
    "PHE":None,
    "PRO":None,
    "SER":None,
    "THR":None,
    "TRP":None,
    "TYR":None,
    "VAL":None,
}

chi4_bond_dict = {
    "ARG":("CD", "NE", ["NH1", "NH2", "CZ"]),
    "LYS":("CD", "CE", ["NZ"]),

}

chi5_bond_dict = {
    "ARG":("NE", "CZ", ["NH1", "NH2"]),
}

complete_chi_bond_dict = {
    "chi1":chi1_bond_dict, "chi2":chi2_bond_dict, "chi3":chi3_bond_dict,
    "chi4":chi4_bond_dict, "chi5":chi5_bond_dict
}

def random_rotate_chi(structure, violation_residue_idx, min_angle=0.6, max_angle=np.pi):
    for res in structure.get_residues():
        if res.get_full_id()[3][1] not in violation_residue_idx:continue
        chi_mask = [0] * 5
        if res.id[0] != " ":
            # print("skip heteroatoms")
            continue
        resname = res.resname
        if resname in ("ALA", "GLY"):
            # print("skip ALA and GLY")
            continue
        for x, chi in enumerate(chi_names):
            chi_res = chi_atoms[chi]
            if resname not in chi_res:
                continue
            chi_mask[x] = 1
        pred_chi = np.random.uniform(min_angle,max_angle,5)
        for i in range(5):
            if chi_mask[i] == 0:
                continue
            chi_bond_dict = eval(f'chi{i+1}_bond_dict')

            atom1, atom2, rotate_atom_list = chi_bond_dict[resname]
            eps = 1e-6
            if (atom1 not in res) or (atom2 not in res):
                continue
            atom1_coord = res[atom1].coord
            atom2_coord = res[atom2].coord
            rot_vec = atom2_coord - atom1_coord
            rot_vec = pred_chi[i] * (rot_vec) / (np.linalg.norm(rot_vec) + eps)
            rot_mat = R.from_rotvec(rot_vec).as_matrix()

            for rotate_atom in rotate_atom_list:
                if rotate_atom not in res:
                    continue
                new_coord = np.matmul(res[rotate_atom].coord - res[atom1].coord, rot_mat.T) + res[atom1].coord
                res[rotate_atom].set_coord(new_coord)
    return structure

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
bond_lengths = {
 ('C', 'CA'): (1.5335,),
 ('C', 'O'): (1.23,),
 ('CA', 'CB'): (1.5365,),
 ('CA', 'N'): (1.4664,),
 ('CB', 'CG'): (1.5331,),
 ('CE', 'SD'): (1.8117,),
 ('CG', 'SD'): (1.8135,),
 ('C', 'N'): (1.3359,),
 ('CB', 'OG'): (1.4144,),
 ('CB', 'SG'): (1.8146,),
 ('CD', 'CG'): (1.5308,),
 ('CD', 'N'): (1.457,),
 ('CB', 'CG1'): (1.5365,),
 ('CB', 'CG2'): (1.5341,),
 ('CD1', 'CG1'): (1.5334,),
 ('CG', 'OD1'): (1.2477,),
 ('CG', 'OD2'): (1.2507,),
 ('CD', 'CE'): (1.5312,),
 ('CE', 'NZ'): (1.4831,),
 ('CD', 'NE'): (1.4748,),
 ('CZ', 'NE'): (1.3205,),
 ('CZ', 'NH1'): (1.3041,),
 ('CZ', 'NH2'): (1.3018,),
 ('CB', 'OG1'): (1.4086,),
 ('CD1', 'CG'): (1.5223,),
 ('CD2', 'CG'): (1.4589,),
 ('CD1', 'CE1'): (1.406,),
 ('CD2', 'CE2'): (1.4063,),
 ('CE1', 'CZ'): (1.406,),
 ('CE2', 'CZ'): (1.4059,),
 ('CD', 'NE2'): (1.3141,),
 ('CD', 'OE1'): (1.2493,),
 ('CD', 'OE2'): (1.2524,),
 ('CG', 'ND2'): (1.3112,),
 ('CD1', 'NE1'): (1.3804,),
 ('CD2', 'CE3'): (1.4115,),
 ('CE2', 'CZ2'): (1.401,),
 ('CE2', 'NE1'): (1.3774,),
 ('CE3', 'CZ3'): (1.4076,),
 ('CH2', 'CZ2'): (1.4046,),
 ('CH2', 'CZ3'): (1.4062,),
 ('CD2', 'NE2'): (1.3971,),
 ('CE1', 'ND1'): (1.3424,),
 ('CE1', 'NE2'): (1.3406,),
 ('CG', 'ND1'): (1.3768,),
 ('CZ', 'OH'): (1.3576,),
 ('C', 'OXT'): (1.2491,),
 ('SG', 'SG'): (2.0324,)}
def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)


def get_all_atoms_protein(pdbFile):
    if pdbFile[-4:] == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdbFile, pdbFile)
    all_atoms = list(s.get_atoms())
    # all_heavy_atoms = [atom for atom in all_atoms if atom.element != 'H' and atom.element != 'D' and atom.id != 'OXT']

    return all_atoms

def get_all_non_hydrogen_atoms_ligand(ligandFile,ref_ligandFile):
    # in sdf format.
    try:
        mol = Chem.MolFromMolFile(ligandFile)
    except:
        mol = Chem.MolFromMolFile(ref_ligandFile)
        mol.RemoveAllConformers()
        mol = AddHs(mol)
        generate_conformer(mol)
    mol = Chem.RemoveAllHs(mol)
    smiles = Chem.MolToSmiles(mol)

    m_order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )
    mol = Chem.RenumberAtoms(mol, m_order)
    mol_atoms = list(mol.GetAtoms())
    mol_atoms = [a.GetSymbol() for a in mol_atoms]

    c = mol.GetConformer()
    mol_atom_coords = c.GetPositions()
    return mol_atom_coords, Chem.GetAdjacencyMatrix(mol).astype(bool)


def compute_local_geometry_violations_protein_v3(ref_proteinFile, proteinFile):
#     ref_protein_all_atoms = get_all_non_hydrogen_atoms_protein(ref_proteinFile)
#     ref_protein_atom_coords = np.array([atom.coord for atom in ref_protein_all_atoms])
    protein_all_atoms = get_all_atoms_protein(proteinFile)
    protein_atom_coords = np.array([atom.coord for atom in protein_all_atoms])
    # print(ref_proteinFile,proteinFile,len(ref_protein_atom_coords) , len(protein_atom_coords))
#     assert len(ref_protein_atom_coords) == len(protein_atom_coords)

    # bonding_distance = 2.0
    # ns = NeighborSearch(ref_protein_all_atoms)
    # bond_list = ns.search_all(bonding_distance)
    #
    # index_dict = {}
    # for i, atom in enumerate(ref_protein_all_atoms):
    #     index_dict[atom.full_id] = i
    #
    # n = len(bond_list)
    # bond_dis_table = np.zeros(n)
    # bond_index_table = []
    # for idx, bond in enumerate(bond_list):
    #     atom1 = bond[0]
    #     atom2 = bond[1]
    #     i = index_dict[atom1.full_id]
    #     j = index_dict[atom2.full_id]
    #     dis = atom1 - atom2
    #     bond_dis_table[idx] = dis
    #     bond_index_table.append([i, j])

    # protein_pdb = openmm_app.PDBFile(proteinFile)
    protein_pdb = openmm_app.PDBxFile(proteinFile) if proteinFile[-4:] == ".cif" else openmm_app.PDBFile(proteinFile)
    bond_index_table = [[bond.atom1.index,bond.atom2.index] for bond in protein_pdb.topology.bonds()]

    violation_residue_idx = []
    deviation_greater_than_cutoff = 0
    for idx, bond_index in enumerate(bond_index_table):
        i, j = bond_index
#         atom1 = ref_protein_all_atoms[i]
#         atom2 = ref_protein_all_atoms[j]

#         ref_dis = atom1 - atom2

        atom1 = protein_all_atoms[i]
        atom2 = protein_all_atoms[j]
        if atom1.id == 'OXT' or atom2.id == 'OXT':continue
        dis = atom1 - atom2
        deviation = abs(bond_lengths[(atom1.name,atom2.name)][0] - dis)
        violation = deviation > 0.3
        deviation_greater_than_cutoff += violation
        if violation:
            violation_residue_idx.extend([atom1.get_full_id()[3][1],atom2.get_full_id()[3][1]])
    return deviation_greater_than_cutoff, violation_residue_idx

def compute_local_geometry_violations_protein(ref_proteinFile, proteinFile):
    ref_protein_all_atoms = get_all_atoms_protein(ref_proteinFile)
    residue_idx = [a.get_full_id()[3][1] for a in ref_protein_all_atoms]
    residue_idx_array = np.array([residue_idx for _ in range(len(residue_idx))])
    ref_protein_atom_coords = np.array([atom.coord for atom in ref_protein_all_atoms])
    protein_all_atoms = get_all_atoms_protein(proteinFile)
    protein_atom_coords = np.array([atom.coord for atom in protein_all_atoms])
    # print(len(ref_protein_atom_coords), len(protein_atom_coords))
    assert len(ref_protein_atom_coords) == len(protein_atom_coords)

    ref_pair_dis = cdist(ref_protein_atom_coords, ref_protein_atom_coords)
    # local geometry
    ref_pair_dis[np.diag_indices(len(ref_pair_dis))] = 10
    local_geometry_mask = ref_pair_dis < 2.0
    pair_dis = cdist(protein_atom_coords, protein_atom_coords)
    deviation_greater_than_cutoff = (abs(ref_pair_dis[local_geometry_mask] - pair_dis[local_geometry_mask]) > 0.3).sum()
    here = np.where(abs(ref_pair_dis[local_geometry_mask] - pair_dis[local_geometry_mask]) > 0.3)
    # print(residue_idx_array[local_geometry_mask][here])
    # local_geometry_score = 1 - (deviation_greater_than_cutoff / local_geometry_mask.sum())
    return deviation_greater_than_cutoff


def compute_local_geometry_violations_ligand(ref_ligandFile, ligandFile):
    # ref_ligandFile = "/gxr/luwei/dynamicbind/run_predictions/dynamicbind_sanyueqi_0516/results/SETD2/index3_idx_3/rank40_ligand_lddt0.50_affinity6.46.sdf"
    # ligandFile = "/gxr/luwei/dynamicbind/run_predictions/dynamicbind_sanyueqi_0516/results/SETD2/index3_idx_3/rank40_ligand_lddt0.50_affinity6.46_relaxed_s_0_ls_0.sdf"

    ref_ligand_atom_coords, local_geometry_mask = get_all_non_hydrogen_atoms_ligand(ref_ligandFile,ligandFile)
    ligand_atom_coords, _ = get_all_non_hydrogen_atoms_ligand(ligandFile,ref_ligandFile)

    assert len(ref_ligand_atom_coords) == len(ligand_atom_coords)

    ref_pair_dis = cdist(ref_ligand_atom_coords, ref_ligand_atom_coords)
    pair_dis = cdist(ligand_atom_coords, ligand_atom_coords)
    deviation_greater_than_cutoff = (abs(ref_pair_dis[local_geometry_mask] - pair_dis[local_geometry_mask]) > 0.3).sum()
    return deviation_greater_than_cutoff

def run_relax(x):
    ref_proteinFile, ref_ligandFile, pdbFile, ligandFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu = x
    try:
        retry = 0
        ret = openmm_relax((pdbFile, ligandFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu))
        while ret['efinal'] > 0 and retry < 5:
            ret = openmm_relax((relaxed_proteinFile, ligandFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu))
            retry += 1
        protein_score, violation_residue_idx = compute_local_geometry_violations_protein_v3(ref_proteinFile, relaxed_proteinFile)
        # print(pdbFile,protein_score)
        retry = 0
        while (protein_score > 0 or ret['efinal'] > 0) and retry < 5:
            parser = MMCIFParser(QUIET=True) if relaxed_proteinFile[-4:] == ".cif" else PDBParser(QUIET=True)
            structure = parser.get_structure("x", relaxed_proteinFile)
            random_rotate_chi(structure, violation_residue_idx, min_angle=-np.pi/2, max_angle=np.pi/2)
            save_protein(structure, relaxed_proteinFile, ca_only=False)
            ret = openmm_relax((relaxed_proteinFile, ligandFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu))
            protein_score, violation_residue_idx = compute_local_geometry_violations_protein_v3(ref_proteinFile, relaxed_proteinFile)
            # print(pdbFile,protein_score)
            retry += 1
        ligand_score = compute_local_geometry_violations_ligand(ref_ligandFile, relaxed_ligandFile)
        # print(protein_score , ligand_score ,ret['efinal'])
        if (protein_score > 0 or ligand_score > 0 or ret['efinal'] > 1000):
            # ret['efinal'] threshould is set to 1000, to allow rare case where protein conformation change of the relaxed structrure is large, but correct.
            print(ref_proteinFile, protein_score , ligand_score ,ret['efinal'])
            os.system(f"rm {pdbFile}")
            os.system(f"rm {ligandFile}")
            os.system(f"rm {relaxed_proteinFile}")
            os.system(f"rm {relaxed_complexFile}")
            os.system(f"rm {relaxed_ligandFile}")
            rank = os.path.basename(pdbFile).split('_')[0]
            # print(f"rm {os.path.dirname(pdbFile)}/{rank}_reverseprocess_data_list.pkl")
            os.system(f"rm {os.path.dirname(pdbFile)}/{rank}_reverseprocess_data_list.pkl")
        os.system(f"rm {fixed_pdbFile}")
        return 0
    except Exception as e:
        print(e)
        os.system(f"rm {pdbFile}")
        os.system(f"rm {ligandFile}")
        rank = os.path.basename(pdbFile).split('_')[0]
        os.system(f"rm {os.path.dirname(pdbFile)}/{rank}_reverseprocess_data_list.pkl")
        return 1


if __name__ == '__main__':
    input_ = []
    idx = 0
    results_path_containments = sorted(os.listdir(args.results_path))
    results_path_containments = [x for x in results_path_containments if x != 'affinity_prediction.csv']

    for rp in results_path_containments:
        if not rp.startswith('index'):
            continue
        # if int(rp.split('_')[0][5:]) > 20:continue
        write_dir = os.path.join(args.results_path, rp)
        file_paths = sorted(os.listdir(write_dir))
        ref_proteinFile = os.path.join(write_dir, [path for path in file_paths if f'ref_proteinFile' in path][0])
        # input structure maybe different with rdkit comformation
        try:
            ref_ligandFile = os.path.join(write_dir, [path for path in file_paths if f'ref_ligandFile' in path][0])
        except:
            ref_ligandFile = ''
        for rank in range(args.samples_per_complex):
            try:
                ligand_file_name = [path for path in file_paths if f'rank{rank+1}_ligand_lddt' in path][0]
                protein_file_name = [path for path in file_paths if f'rank{rank+1}_receptor_lddt' in path][0]
            except:
                continue
            pdb_or_cif = protein_file_name[-3:]
            pdbFile = os.path.join(write_dir, protein_file_name)
            ligandFile = os.path.join(write_dir, ligand_file_name)
            fixed_pdbFile = os.path.join(write_dir, f'fixed_{idx}.{pdb_or_cif}')
            relaxed_proteinFile = os.path.join(write_dir, protein_file_name.replace(f'.{pdb_or_cif}',f'_relaxed.{pdb_or_cif}'))
            gap_mask = "none"
            stiffness, ligand_stiffness = 1000, 3000
            relaxed_complexFile = relaxed_proteinFile.replace("_receptor_", "_complex_")
            relaxed_ligandFile = os.path.join(write_dir, ligand_file_name.replace('.sdf','_relaxed.sdf'))
            use_gpu = True
            x = (ref_proteinFile, ref_ligandFile, pdbFile, ligandFile, fixed_pdbFile, relaxed_proteinFile, gap_mask, stiffness, ligand_stiffness, relaxed_complexFile, relaxed_ligandFile, use_gpu)
            input_.append(x)
            idx += 1
    # print(input_)
    # raise
    # for x in input_:
    #     print(x[0],x[2])
    # raise
    r = process_map(run_relax, input_, max_workers=args.num_workers)

    for rp in results_path_containments:
        if not rp.startswith('index'):
            continue
        # if int(rp.split('_')[0][5:]) > 20:continue
        write_dir = os.path.join(args.results_path, rp)
        file_paths = sorted(os.listdir(write_dir))
        orig_rank = [int(fn.split('_')[0][4:]) for fn in file_paths if 'reverseprocess_data_list.pkl' in fn]
        for i,rank in enumerate(sorted(orig_rank)):
            ligand_file_name = [path for path in file_paths if f'rank{rank}_ligand_lddt' in path and 'relaxed' not in path][0]
            protein_file_name = [path for path in file_paths if f'rank{rank}_receptor_lddt' in path and 'relaxed' not in path][0]
            relaxed_ligand_file_name = [path for path in file_paths if f'rank{rank}_ligand_lddt' in path and 'relaxed' in path][0]
            relaxed_protein_file_name = [path for path in file_paths if f'rank{rank}_receptor_lddt' in path and 'relaxed' in path][0]
            relaxed_complex_file_name = relaxed_protein_file_name.replace("_receptor_", "_complex_")
            data_file_name = f'rank{rank}_reverseprocess_data_list.pkl'
            new_ligand_file_name = ligand_file_name.replace(f'rank{rank}',f'rank{i+1}')
            new_protein_file_name = protein_file_name.replace(f'rank{rank}',f'rank{i+1}')
            new_relaxed_ligand_file_name = relaxed_ligand_file_name.replace(f'rank{rank}',f'rank{i+1}')
            new_relaxed_protein_file_name = relaxed_protein_file_name.replace(f'rank{rank}',f'rank{i+1}')
            new_relaxed_complex_file_name = relaxed_complex_file_name.replace(f'rank{rank}',f'rank{i+1}')
            new_data_file_name = data_file_name.replace(f'rank{rank}',f'rank{i+1}')
            os.rename(f"{os.path.join(write_dir, ligand_file_name)}",f"{os.path.join(write_dir, new_ligand_file_name)}")
            os.rename(f"{os.path.join(write_dir, protein_file_name)}",f"{os.path.join(write_dir, new_protein_file_name)}")
            os.rename(f"{os.path.join(write_dir, relaxed_ligand_file_name)}",f"{os.path.join(write_dir, new_relaxed_ligand_file_name)}")
            os.rename(f"{os.path.join(write_dir, relaxed_protein_file_name)}",f"{os.path.join(write_dir, new_relaxed_protein_file_name)}")
            os.rename(f"{os.path.join(write_dir, data_file_name)}",f"{os.path.join(write_dir, new_data_file_name)}")
            os.rename(f"{os.path.join(write_dir, relaxed_complex_file_name)}",f"{os.path.join(write_dir, new_relaxed_complex_file_name)}")
    # with Pool(args.num_workers) as p:
    #     r = list(tqdm.tqdm(p.imap(run_relax, input_), total=len(input_)))
