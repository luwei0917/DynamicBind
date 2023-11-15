import numpy as np
import pandas as pd

import os
import sys
import rdkit.Chem as Chem

from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch
from Bio.PDB import Superimposer
from Bio.PDB import PDBIO, Select

from scipy.spatial.distance import cdist

from argparse import ArgumentParser

def get_all_non_hydrogen_atoms_protein(pdbFile):
    if pdbFile[-4:] == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdbFile, pdbFile)
    all_atoms = list(s.get_atoms())
    all_heavy_atoms = [atom for atom in all_atoms if atom.element != 'H' and atom.element != 'D' and atom.id != 'OXT']
    
    return all_heavy_atoms

def get_all_non_hydrogen_atoms_ligand(ligandFile):
    # in sdf format.
    mol = Chem.MolFromMolFile(ligandFile)
    mol_atoms = list(mol.GetAtoms())
    mol_atoms = [a.GetSymbol() for a in mol_atoms]

    c = mol.GetConformer()
    mol_atom_coords = c.GetPositions()
    return mol_atom_coords, Chem.GetAdjacencyMatrix(mol).astype(bool)



def compute_local_geometry_violations_protein(ref_proteinFile, proteinFile):
    ref_protein_all_atoms = get_all_non_hydrogen_atoms_protein(ref_proteinFile)
    ref_protein_atom_coords = np.array([atom.coord for atom in ref_protein_all_atoms])
    protein_all_atoms = get_all_non_hydrogen_atoms_protein(proteinFile)
    protein_atom_coords = np.array([atom.coord for atom in protein_all_atoms])
    assert len(ref_protein_atom_coords) == len(protein_atom_coords)

    ref_pair_dis = cdist(ref_protein_atom_coords, ref_protein_atom_coords)
    # local geometry
    ref_pair_dis[np.diag_indices(len(ref_pair_dis))] = 10
    local_geometry_mask = ref_pair_dis < 2.0
    pair_dis = cdist(protein_atom_coords, protein_atom_coords)
    deviation_greater_than_cutoff = (abs(ref_pair_dis[local_geometry_mask] - pair_dis[local_geometry_mask]) > 0.3).sum()
    # local_geometry_score = 1 - (deviation_greater_than_cutoff / local_geometry_mask.sum())
    return deviation_greater_than_cutoff



def compute_local_geometry_violations_protein_v2(ref_proteinFile, proteinFile):
    ref_protein_all_atoms = get_all_non_hydrogen_atoms_protein(ref_proteinFile)
    ref_protein_atom_coords = np.array([atom.coord for atom in ref_protein_all_atoms])
    protein_all_atoms = get_all_non_hydrogen_atoms_protein(proteinFile)
    protein_atom_coords = np.array([atom.coord for atom in protein_all_atoms])
    assert len(ref_protein_atom_coords) == len(protein_atom_coords)

    bonding_distance = 2.0
    ns = NeighborSearch(ref_protein_all_atoms)
    bond_list = ns.search_all(bonding_distance)

    index_dict = {}
    for i, atom in enumerate(ref_protein_all_atoms):
        index_dict[atom.full_id] = i

    n = len(bond_list)
    bond_dis_table = np.zeros(n)
    bond_index_table = []
    for idx, bond in enumerate(bond_list):
        atom1 = bond[0]
        atom2 = bond[1]
        i = index_dict[atom1.full_id]
        j = index_dict[atom2.full_id]
        dis = atom1 - atom2
        bond_dis_table[idx] = dis
        bond_index_table.append([i, j])

    deviation_greater_than_cutoff = 0
    for idx, bond_index in enumerate(bond_index_table):
        i, j = bond_index
        atom1 = protein_all_atoms[i]
        atom2 = protein_all_atoms[j]
        dis = atom1 - atom2
        deviation = abs(bond_dis_table[idx] - dis)
        deviation_greater_than_cutoff += deviation > 0.3
    return deviation_greater_than_cutoff


def compute_local_geometry_violations_ligand(ref_ligandFile, ligandFile):
    # ref_ligandFile = "/gxr/luwei/dynamicbind/run_predictions/dynamicbind_sanyueqi_0516/results/SETD2/index3_idx_3/rank40_ligand_lddt0.50_affinity6.46.sdf"
    # ligandFile = "/gxr/luwei/dynamicbind/run_predictions/dynamicbind_sanyueqi_0516/results/SETD2/index3_idx_3/rank40_ligand_lddt0.50_affinity6.46_relaxed_s_0_ls_0.sdf"
    
    ref_ligand_atom_coords, local_geometry_mask = get_all_non_hydrogen_atoms_ligand(ref_ligandFile)
    ligand_atom_coords, _ = get_all_non_hydrogen_atoms_ligand(ligandFile)

    assert len(ref_ligand_atom_coords) == len(ligand_atom_coords)

    ref_pair_dis = cdist(ref_ligand_atom_coords, ref_ligand_atom_coords)
    pair_dis = cdist(ligand_atom_coords, ligand_atom_coords)
    deviation_greater_than_cutoff = (abs(ref_pair_dis[local_geometry_mask] - pair_dis[local_geometry_mask]) > 0.3).sum()
    return deviation_greater_than_cutoff



parser = ArgumentParser(description="Check violations. example: python check_structure_violations.py /mnt/nas/research-data/luwei/dynamicbind_data/database/wholePDB_v3//pocket_aligned_fill_missing/Q9BYW2/af2_7ty2_KS6_A_aligned.pdb --proteinFile /gxr/luwei/dynamicbind/run_predictions/dynamicbind_sanyueqi_0516/results/SETD2/back_index3_idx_3/rank20_receptor_lddt0.54_affinity6.85_relaxed.pdb ")
parser.add_argument('reference', type=str, help='reference, could be protein in pdb or cif formation, or ligand in sdf formation.')
parser.add_argument('-p', '--proteinFile', type=str, default=None)
parser.add_argument('-l', '--ligandFile', type=str, default=None, help='in sdf format')

args = parser.parse_args()

if (args.proteinFile is None) and (args.ligandFile is None):
    raise("need either protein file or ligand file")
if args.proteinFile is not None:
    score = compute_local_geometry_violations_protein_v2(args.reference, args.proteinFile)
    if score > 0:
        print("Has violations", score)
    else:
        print("No violations.")
if args.ligandFile is not None:
    score = compute_local_geometry_violations_ligand(args.reference, args.ligandFile)
    if score > 0:
        print("Has violations", score)
    else:
        print("No violations.")