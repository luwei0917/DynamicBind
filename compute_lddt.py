import pandas as pd
import numpy as np
import os
import sys

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import Superimposer
from Bio.PDB import PDBIO, Select
from Bio.PDB.Polypeptide import protein_letters_3to1
from scipy.spatial.distance import cdist
from collections import defaultdict

import rdkit.Chem as Chem

from helper_functions import gap_mask_all_res, align_to_original

ambiguous_atom_dict = defaultdict(lambda : 0)
# It has to be these seven amino acid, to match exactly with the lDDT software.
ambiguous_atom_dict.update({
    "ASP_OD1":1,
    "ASP_OD2":2,
    "GLU_OE1":1,
    "GLU_OE2":2,
    "PHE_CD1":1,
    "PHE_CD2":2,
    "PHE_CE1":1,
    "PHE_CE2":2,
    "TYR_CD1":1,
    "TYR_CD2":2,
    "TYR_CE1":1,
    "TYR_CE2":2,
    "LEU_CD1":1,
    "LEU_CD2":2,
    "VAL_CG1":1,
    "VAL_CG2":2,
    "ARG_NH1":1,
    "ARG_NH2":2,
})
# LEU, VAL and ARG
def get_mapped_atom_coords(ref_all_res, all_res, ambiguous_atom_dict=ambiguous_atom_dict):
    ref_atom_coord_list = []
    atom_coord_list = []
    res_idx_list = []
    skip_atom_count = 0
    ambiguous_atom_list = []
    for i, (ref_res, res) in enumerate(zip(ref_all_res, all_res)):
        # assert ref_res.resname == res.resname
        for ref_atom in ref_res:
            if ref_atom.id in res:
                atom = res[ref_atom.id]
                ref_atom_coord_list.append(ref_atom.coord)
                atom_coord_list.append(atom.coord)
                res_idx_list.append(i)
                ambiguous_atom_list.append(ambiguous_atom_dict[f"{ref_res.resname}_{ref_atom.id}"])
            elif ref_atom.element == 'H' or ref_atom.element == 'D':
                # D is Deuterium
                pass
            else:
                skip_atom_count += 1
                pass
                # print(ref_atom)
    if skip_atom_count > 10:
        print("skipped more than 10 atoms, sometime might be wrong.")
    ref_atom_coords = np.array(ref_atom_coord_list)
    atom_coords = np.array(atom_coord_list)
    return ref_atom_coords, atom_coords, np.array(res_idx_list), np.array(ambiguous_atom_list)

def swap_ambiguous(res_level_deviation, res_ambiguous):
    res_level_deviation = res_level_deviation.copy()
    res_level_deviation[res_ambiguous == 1], res_level_deviation[res_ambiguous == 2] = res_level_deviation[res_ambiguous == 2], res_level_deviation[res_ambiguous == 1]
    sawpped_res_ambiguous = res_ambiguous.copy()
    sawpped_res_ambiguous[res_ambiguous==1], sawpped_res_ambiguous[res_ambiguous==2] = sawpped_res_ambiguous[res_ambiguous==2], sawpped_res_ambiguous[res_ambiguous==1]
    return res_level_deviation, sawpped_res_ambiguous

def compute_conserved_distances(deviation, mask, thresholds=[0.5, 1, 2, 4]):
    conserved_distances = 0
    for thr in thresholds:
        conserved_distances += ((deviation <= thr) * mask).sum()
    return conserved_distances


def find_chosen_atom_idx_list(deviation, swapped_deviation, mask, 
                              ambiguous_atom_list, atom_idx_list, swapped_atom_idx_list, res_idx_list):
    # find the best atom idx for each residue.
    # deviation for residue i with all non-ambiguous atom
    deviation_with_non_ambiguous_atom = deviation[:, ambiguous_atom_list == 0]
    swapped_deviation_with_non_ambiguous_atom = swapped_deviation[:, ambiguous_atom_list == 0]
    mask_with_non_ambiguous_atom = mask[:, ambiguous_atom_list == 0]

    # idx = 3
    chosen_atom_idx_list = np.zeros_like(atom_idx_list)
    for idx in range(res_idx_list[-1]+1):
        this_res_idx = res_idx_list == idx
        res_level_deviation = deviation_with_non_ambiguous_atom[this_res_idx]
        res_level_mask = mask_with_non_ambiguous_atom[this_res_idx]
        res_ambiguous = ambiguous_atom_list[this_res_idx]
        res_lddt = compute_conserved_distances(res_level_deviation, res_level_mask)
        if (res_ambiguous != 0).sum() > 0:
            swapped_res_level_deviation = swapped_deviation_with_non_ambiguous_atom[this_res_idx]
            swapped_res_lddt = compute_conserved_distances(swapped_res_level_deviation, res_level_mask)
            if swapped_res_lddt > res_lddt:
                chosen_atom_idx_list[this_res_idx] = swapped_atom_idx_list[this_res_idx]
            else:
                chosen_atom_idx_list[this_res_idx] = atom_idx_list[this_res_idx]
        else:
            chosen_atom_idx_list[this_res_idx] = atom_idx_list[this_res_idx]
    return chosen_atom_idx_list

def get_matched_res(aligned_result, all_res, ref_all_res):
    ref_idx = mod_idx = 0
    matched_all_res = []
    ref_matched_all_res = []
    for ref, mod in zip(*aligned_result):
        if ref == '-':
            mod_idx += 1
        elif mod == '-':
            ref_idx += 1
        elif ref != mod:
            mod_idx += 1
            ref_idx += 1
        else:
            matched_all_res.append(all_res[mod_idx])
            ref_matched_all_res.append(ref_all_res[ref_idx])
            mod_idx += 1
            ref_idx += 1
    return ref_matched_all_res, matched_all_res


def create_swapped_atom_idx_list(atom_idx_list, ambiguous_atom_list, res_idx_list, verbose=False):
    swapped_atom_idx_list = np.arange(len(ambiguous_atom_list))
    for idx in range(res_idx_list[-1]+1):
        this_res_idx = res_idx_list == idx
        this_ambiguous_atom_list = ambiguous_atom_list[this_res_idx]
        ambiguous_count = (this_ambiguous_atom_list != 0).sum()
        if ambiguous_count == 0:
            pass
        if (this_ambiguous_atom_list == 1).sum() != (this_ambiguous_atom_list == 2).sum():
            if verbose:
                print(f"res idx {idx} contains missing residue")
        else:
            swapped_atom_idx_list[atom_idx_list[this_res_idx][this_ambiguous_atom_list==1]] = atom_idx_list[this_res_idx][this_ambiguous_atom_list==2]
            swapped_atom_idx_list[atom_idx_list[this_res_idx][this_ambiguous_atom_list==2]] = atom_idx_list[this_res_idx][this_ambiguous_atom_list==1]
    return swapped_atom_idx_list


def compute_lddt(modelFile, refFile, per_res=None, binding_site=None, 
                 need_alignment=True, inclusion_radius=15, seq_sep=0, 
                 binding_site_cutoff=4.0, verbose=False, gap_mask=None):
    if modelFile[-4:] == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    s_model = parser.get_structure("x", modelFile)
    chains = list(s_model.get_chains())
    if len(chains) > 1:
        if verbose:
            print("has more than one chain, pick the first one")
    c = s_model[0][chains[0].id]
    all_res = gap_mask_all_res(list(c.get_residues()), gap_mask)
    
    model_seq = "".join([protein_letters_3to1[res.resname] for res in all_res])
    # model_seq
    if refFile[-4:] == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    s_ref = parser.get_structure("x", refFile)
    ref_chains = list(s_ref.get_chains())
    if len(ref_chains) > 1:
        if verbose:
            print("has more than one chain, pick the first one")
    ref_c = s_ref[0][ref_chains[0].id]
    ref_all_res = gap_mask_all_res(list(ref_c.get_residues()), gap_mask)
    ref_model_seq = "".join([protein_letters_3to1[res.resname] for res in ref_all_res])
    # ref_model_seq
    if need_alignment:
        aligned_result = align_to_original(ref_model_seq, model_seq)
        ref_all_res, all_res = get_matched_res(aligned_result, all_res, ref_all_res)
    ref_atom_coords, atom_coords, res_idx_list, ambiguous_atom_list = get_mapped_atom_coords(ref_all_res, all_res)
    
    pair_dis = cdist(atom_coords, atom_coords)
    ref_pair_dis = cdist(ref_atom_coords, ref_atom_coords)


    dis_mask = ref_pair_dis < inclusion_radius
    res_gap = (res_idx_list[None,] - res_idx_list[:,None])

    gap_mask = abs(res_gap) > seq_sep
    mask = dis_mask & gap_mask

    if (ambiguous_atom_list == 1).sum() != (ambiguous_atom_list == 2).sum():
        if verbose:
            print("probably contains missing atoms.", modelFile, refFile)

    atom_idx_list = np.arange(len(ambiguous_atom_list))
    swapped_atom_idx_list = np.arange(len(ambiguous_atom_list))
    swapped_atom_idx_list = create_swapped_atom_idx_list(atom_idx_list, ambiguous_atom_list, res_idx_list)
    # swapped_atom_idx_list[ambiguous_atom_list == 1], swapped_atom_idx_list[ambiguous_atom_list == 2] = atom_idx_list[ambiguous_atom_list == 2], atom_idx_list[ambiguous_atom_list == 1]

    # swapped_pair_dis = cdist(atom_coords[swapped_atom_idx_list], atom_coords[swapped_atom_idx_list])
    # abs(pair_dis[swapped_atom_idx_list][:, swapped_atom_idx_list] - swapped_pair_dis).sum()
    swapped_pair_dis = pair_dis[swapped_atom_idx_list][:, swapped_atom_idx_list]
    deviation = abs(pair_dis - ref_pair_dis)
    swapped_deviation = abs(swapped_pair_dis - ref_pair_dis)

    chosen_atom_idx_list = find_chosen_atom_idx_list(deviation, swapped_deviation, mask, 
                                            ambiguous_atom_list, atom_idx_list, swapped_atom_idx_list, res_idx_list)

    chosen_pair_dis = pair_dis[chosen_atom_idx_list][:, chosen_atom_idx_list]
    chosen_deviation = abs(chosen_pair_dis - ref_pair_dis)
    
    lddt_mask = np.triu(mask)
    complete_conserved_distances = compute_conserved_distances(chosen_deviation, lddt_mask)
    complete_total_n_distances = lddt_mask.sum() * 4

    if per_res is not None:
        info = []
        for idx in range(res_idx_list[-1]+1):
            res_level_deviation = chosen_deviation[res_idx_list == idx]
            res_level_mask = mask[res_idx_list == idx]
            conserved_distances = compute_conserved_distances(res_level_deviation, res_level_mask)
            total_n_distances = res_level_mask.sum() * 4
            res_score = round(conserved_distances/total_n_distances, 4)
            info.append([res_score, conserved_distances, total_n_distances])
        info = pd.DataFrame(info, columns=['Score', 'Conserved', 'Total'])
    else:
        info = None
    result = {"lddt":round(complete_conserved_distances/complete_total_n_distances, 4),
             "conserved_distances":complete_conserved_distances,
             "total_n_distances":complete_total_n_distances,
             "per_res_lddt":info}

    if binding_site is not None:
        ligand_atom_coords = binding_site
        chosen_atom_coords = atom_coords[chosen_atom_idx_list]
        protein_ligand_dis = cdist(chosen_atom_coords, ligand_atom_coords)
        binding_site_mask = protein_ligand_dis.min(axis=-1) < binding_site_cutoff
        if binding_site_mask.sum() == 0:
            if verbose:
                print("no protein atom is within cutoff from ligand atoms")
            result.update({
                "bs_lddt":0,
                "binding_site_conserved_distances":0,
                "binding_site_total_n_distances":0,
            })
            return result
        binding_site_lddt_mask = np.triu(mask[binding_site_mask][:, binding_site_mask])
        binding_site_chosen_deviation = chosen_deviation[binding_site_mask][:, binding_site_mask]

        binding_site_conserved_distances = compute_conserved_distances(binding_site_chosen_deviation, binding_site_lddt_mask)
        binding_site_total_n_distances = binding_site_lddt_mask.sum() * 4
        result.update({
            "bs_lddt":round(binding_site_conserved_distances / (1e-5 + binding_site_total_n_distances), 4),
            "binding_site_conserved_distances":binding_site_conserved_distances,
            "binding_site_total_n_distances":binding_site_total_n_distances,
        })
    return result
