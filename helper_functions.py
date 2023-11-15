import requests as r
from Bio import SeqIO
from io import StringIO
import re
import pandas as pd
import numpy as np
from Bio import SeqIO
import uuid
import copy
import os
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import Superimposer
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB import MMCIFIO
from tqdm import tqdm

import rdkit.Chem as Chem
from rdkit.Geometry import Point3D

from typing import Optional
import requests
import json


three_to_index = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7, 
                'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15, 
                'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19}
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


def shift_to_end(data, col_name):
    col = data.pop(col_name)
    data[col_name] = col

def shift_to_front(d, cols_to_front):
    # cols_to_front = ['rank', 'hasStar']
    cols = d.columns
    cols = cols_to_front + [col for col in cols if col not in cols_to_front]
    d = d[cols]
    return d

def get_url(uniprot_id):
    currentUrl = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    response = r.get(currentUrl)
    cData=''.join(response.text)
    return cData

def get_cath_group(uniprot_id, cData=None):
    if cData is None:
        cData = get_url(uniprot_id)
    m = re.search(r'Gene3D; ([.\d]*);', cData)
    try:
        cath_group = m.groups()[0]
    except Exception as e:
        # print(uniprot_id, "error", e)
        return ""
    return cath_group

def get_ec_group(uniprot_id, cData=None):
    if cData is None:
        cData = get_url(uniprot_id)
    m = re.search('EC=([.\d]*);', cData)
    try:
        ec_group = m.groups()[0]
    except Exception as e:
        # print(uniprot_id, "error", e)
        return ""
    return ec_group

def get_ec_group(uniprot_id, cData=None):
    if cData is None:
        cData = get_url(uniprot_id)
    m = re.search('EC=([.\d-]*)', cData)
    try:
        ec_group = m.groups()[0]
    except Exception as e:
        # print(uniprot_id, "error", e)
        return ""
    return ec_group

def remove_alternative_atom_coordniates(fileName, toFileName):
    # fileName = "/gxr/luwei/covid/docking/7B3O/ab.pdb"
    # toFileName = "/gxr/luwei/covid/docking/7B3O/ab_fixed.pdb"
    # remove alternative atom coordinates.
    # change A to default.
    # for example
    # ATOM    267  CA AASN E 354     -26.316 -17.365  11.816  0.51 34.75           C  
    # ATOM    268  CA BASN E 354     -26.315 -17.367  11.818  0.49 34.75           C 
    with open(fileName) as f:
        a = f.readlines()
    with open(toFileName, "w") as out:
        for line in a:
            if len(line) < 17:
                out.write(line)
                continue
            if line[16] == 'A':
                b = list(line)
                b[16] = " "
                out.write("".join(b))
            elif line[16] == ' ':
                out.write(line)
            else:
                pass


def read_fasta(fastaFile):
    record = SeqIO.read(fastaFile, "fasta")
    seq = str(record.seq)
    return seq

def read_fasta_v2(fastaFile):
    for record in SeqIO.parse(fastaFile, "fasta"):
        seq = str(record.seq)
        break
    return seq

def read_fasta_v3(fastaFile):
    result = []
    for record in SeqIO.parse(fastaFile, "fasta"):
        seq = str(record.seq)
        result.append([record.id, seq])
    result = pd.DataFrame(result, columns=['id', 'seq'])
    return result
    
def align_to_original(seq_origin, seq_new):
    unique_filename = str(uuid.uuid4())
    tmp_fasta = f"./{unique_filename}.fasta"
    seq_list = [seq_origin, seq_new]
    with open(tmp_fasta, "w") as f:
        for i, seq in enumerate(seq_list):
            f.write(f">{i}\n")
            f.write(f"{seq}\n")
    r = os.popen(f"/gxr/luwei/anaconda3/envs/py38/bin/kalign {tmp_fasta}")
    results = r.readlines()
    os.system(f"rm {tmp_fasta}")
    # print(results)
    out_list = []
    seq = ""
    for line in results:
        if line[0] == ">":
            out_list.append(seq)
            seq = ""
        else:
            seq += line.strip()
    out_list.append(seq)
    out_list = out_list[1:]
    return out_list
# parser = PDBParser(QUIET=True)

# def get_atom_list(res_list):
#     # get all atoms in res_list
#     atom_list = []
#     for res in res_list:
#         atom_list += list(res.get_atoms())
#     return atom_list

def remove_hetero(res_list, verbose=True, ensure_ca_exist=False):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if (not ensure_ca_exist) or ('CA' in res):
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero, removed")
    return clean_res_list
# def extract_resid(res_list):
#     # extract resid information
#     resid_list = []
#     for res in res_list:
#         hetero, resid, insertion = res.full_id[-1]
#         assert hetero == ' '
#         if insertion == ' ':
#             insertion = ''
#         resid_list.append(str(resid)+" "+insertion)
#     return resid_list

def remove_hetero_v2(res_list, verbose=True, ensure_ca_exist=False, bfactor_cutoff=None):
    # could also filter by bfactor.
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero, removed")
    return clean_res_list

def get_aligned_index_info(result, columns=['pdb_seq', 'pdb_idx', 'uniprot_seq', 'uniprot_idx']):
    idx_a = idx_b = 0
    info = []
    for a,b in zip(*result):
        if a == '-' and  b != '-':
            idx_b += 1
            continue
        elif b == '-' and a != '-':
            idx_a += 1
            continue
        elif a != '-' and b != '-':
            idx_a += 1
            idx_b += 1
        else:
            print("error?")
            break
        info.append([a, idx_a-1, b, idx_b-1])
    info = pd.DataFrame(info, columns=columns)
    return info

def extract_pdb_based_on_p2rank(uid, p2rankFile, pdbFile, toFile, radius=20, top_n_pocket=1):
    d = pd.read_csv(p2rankFile)
    d.columns = d.columns.str.strip()
    s = parser.get_structure(uid, pdbFile)
    all_res = [res for res in s.get_residues() if res.full_id[3][0]==' ']
    all_cb = [res['CB'] if res.resname != 'GLY' else res['CA'] for res in all_res]

    coord = np.array([atom.coord for atom in all_cb])
    center = d[["center_x", "center_y", "center_z"]].values[:top_n_pocket]

    dis = distance_matrix(center, coord)
    # all_chosen_index = np.argsort(dis, axis=1)[:,:100]
    all_chosen_index = dis < radius
    for i_th_pocket in range(all_chosen_index.shape[0]):
        chosen_index = all_chosen_index[i_th_pocket]
        with open(toFile.replace("ITHPOCKET", str(i_th_pocket)), "w") as f:
            with open(pdbFile) as pdb:
                for line in pdb:
                    if line[:4] == 'ATOM':
                        chain = line[21]
                        res_idx = int(line[22:26])
                        if chain == 'A' and chosen_index[res_idx-1]:
                            # print(line[:-1])
                            f.write(line)
                    else:
                        continue
                        print(line[:-1]) # alternatively, skip with continue


def compute_com_of_ligand(fileName):
    elements_dict = {'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,\
                    'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                    'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
                    'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                    'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
                    'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
                    'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
                    'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
                    'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
                    'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
                    'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
                    'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
                    'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
                    'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
                    'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
                    'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
                    'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
                    'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
                    'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
                    'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
                    'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
                    'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
                    'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
                    'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
                    'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
                    'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
                    'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
                    'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
                    'OG' : 294}

    try:
        mol = pd.read_csv(fileName, skiprows=4, names=['x', 'y', 'z', 'symbol', 'n0', 'n1', 'n2', 'n3', 'n4'], sep='\s+')
        mol = mol.dropna()
        x = mol[['x', 'y', 'z']].values.astype(float)
        atom_mass = np.array([elements_dict[symbol.upper()] for symbol in mol['symbol'].values])
        com = (x * atom_mass.reshape(-1, 1) ).sum(axis=0) / (atom_mass.sum())
    except:
        mol = pd.read_csv(fileName, skiprows=4, names=['x', 'y', 'z', 'symbol', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5'], sep='\s+')
        mol = mol.dropna()
        x = mol[['x', 'y', 'z']].values.astype(float)
        atom_mass = np.array([elements_dict[symbol.upper()] for symbol in mol['symbol'].values])
        com = (x * atom_mass.reshape(-1, 1) ).sum(axis=0) / (atom_mass.sum())
    return com



def read_sw_file(file_loc):
    f = open(file_loc)
    all_lines = f.readlines()
    f.close()
    i = -1
    result = []
    for line in all_lines:
        suboptimal_alignment_score = 0
        i += 1
        if i%4 == 0:
            if 'target_name' in line:
                target_name = line.strip().split(' ')[-1]
            else:
                print("error", line)
        elif i%4 == 1:
            if 'query_name' in line:
                query_name = line.strip().split(' ')[-1]
                #print('query_name',query_name)
            else:
                print("error query", line)
        elif i%4 == 2:
            if 'optimal_alignment_score' in line:

                for item in line.strip().split('\t'):
                    if item.split(' ')[0] == 'optimal_alignment_score:':
                        optimal_alignment_score = float(item.split(' ')[1])
                    if item.split(' ')[0] == 'suboptimal_alignment_score:':
                        suboptimal_alignment_score = int(item.split(' ')[1])
                    elif item.split(' ')[0] == 'target_end:':
                        target_end = int(item.split(' ')[1])
            else:
                print("error score", line)
        else:
            result.append([target_name, query_name, optimal_alignment_score])
    return result

def read_in_normalized_sw_table(self_n, self_m, pairwise, index_table, query_index_table):
    n = len(self_n)
    m = len(self_m)
    target = np.zeros(m)
    for line in self_m:
        target_name, query_name, optimal_alignment_score = line
        target_index = index_table[target_name.split("_")[-1]] + int(target_name.split("_")[-2])
        target[target_index] = optimal_alignment_score

    query = np.zeros(n)
    for line in self_n:
        target_name, query_name, optimal_alignment_score = line
        query_index = query_index_table[query_name.split("_")[-1]] + int(query_name.split("_")[-2])
        query[query_index] = optimal_alignment_score

    sw_score_table = np.zeros((n,m))

    for line in pairwise:
        target_name, query_name, optimal_alignment_score = line
        target_index = index_table[target_name.split("_")[-1]] + int(target_name.split("_")[-2])
        query_index = query_index_table[query_name.split("_")[-1]] + int(query_name.split("_")[-2])
        sw_score_table[query_index, target_index] = optimal_alignment_score

    normalized_sw_table = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            normalized_sw_table[i,j] = sw_score_table[i,j] / np.sqrt(query[i] * target[j])

    return normalized_sw_table



def get_resdiue_cb(res):
    if 'CB' in res:
        return res['CB']
    else:
        return res['CA']

def get_chain_closest_to_ligand_com(all_res, com):
    dis_list = []
    for res in all_res:
        cb = get_resdiue_cb(res)
        dis = (((cb.coord - com)**2).sum())**0.5
        dis_list.append(dis)

    chain_list = [res.full_id[2] for res in all_res]
    chain = chain_list[np.argmin(dis_list)]
    return chain

def get_all_ca(res_list):
    atom_list = []
    for res in res_list:
        atom_list.append(res['CA'])
    return atom_list

# def align_pdb_to_alphaFold_structure(alphaFold, pdbFile, aligned_pdb, name=None, chain=None, pred_chain=None):
#     # Q99683_3vw6 is not correct.
#     # we need align to a single chain, instead of trying to align to multiple.
#     # but it has to be the chain that has the ligand.
#     # we have the COM of the ligand. use this to find the closest chain in pdb.
#     # and we align to this chain.
#     super_imposer = Superimposer()
#     parser = PDBParser(QUIET=True)
#     # fileName = "/gxr/luwei/hashcpi/extract_based_on_p2rank/5ime/5ime_protein.pdb"
#     ref_pdb = parser.get_structure("x", pdbFile)
#     if chain:
#         ref_pdb = ref_pdb[0][chain]
#     all_res = remove_hetero(ref_pdb.get_residues(), verbose=False, ensure_ca_exist=True)
#     pdb_seq = "".join([three_to_one.get(res.resname) for res in all_res])

#     # fileName = "/gxr/luwei/hashcpi/alphafold_pdb_v2/Q13153.pdb"
#     pred_pdb = parser.get_structure("pred", alphaFold)
#     if pred_chain:
#         pred_pdb = pred_pdb[0][pred_chain]
#     pred_all_res = remove_hetero(pred_pdb.get_residues(), verbose=False, ensure_ca_exist=True)
#     pred_seq = "".join([three_to_one.get(res.resname) for res in pred_all_res])
#     # seq = read_fasta("/gxr/luwei/hashcpi/fasta/Q13153.fasta")
#     # assert pred_seq == seq
#     result = align_to_original(pdb_seq, pred_seq)
#     # remove all dashes in pdb_seq
#     info = get_aligned_index_info(result)
#     info = info.query("pdb_seq == uniprot_seq").reset_index(drop=True)
#     identity_ratio = (len(info) / (min(len(pdb_seq), len(pred_seq))))

#     pdb_ca_list = get_all_ca(all_res)
#     pred_ca_list = get_all_ca(pred_all_res)
#     chosen_ca_list = [ca for i, ca in enumerate(pdb_ca_list) if i in info.pdb_idx.values]
#     chosen_pred_ca_list = [ca for i, ca in enumerate(pred_ca_list) if i in info.uniprot_idx.values]
#     super_imposer.set_atoms(chosen_pred_ca_list, chosen_ca_list)
#     super_imposer.apply(ref_pdb.get_atoms())
#     io = PDBIO()
#     io.set_structure(ref_pdb)
#     # io.save("/gxr/luwei/hashcpi/extract_based_on_p2rank/5ime/new_Q13153.pdb")
#     io.save(aligned_pdb)
#     # if identity_ratio < 0.95:
#     if name:
#         result = {'alphaFold':alphaFold,
#                  'pdbFile':pdbFile,
#                  'aligned_pdb':aligned_pdb, 
#                  'identity_ratio':identity_ratio, 'len_info':len(info), 
#                   'len_pdb_seq':len(pdb_seq), 'len_pred_seq':len(pred_seq),
#                 'rotrain':super_imposer.rotran}
#         np.save(name, result)


# shift columns to the end
def shift_col(d, col_to_the_end=None):
    if col_to_the_end:
        # col_to_the_end = ['loss', 'BCEloss', 'precision_0', 'recall_0', 'f1_0']
        col_list = d.columns
        new_col_list = []
        for col in col_list:
            if col in col_to_the_end:
                continue
            new_col_list.append(col)
        new_col_list = new_col_list + col_to_the_end
    return d[new_col_list]

# taken from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
# "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and Blostein, S. D, IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987
# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B, correct_reflection=True):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0 and correct_reflection:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

# def compute_rmsd_np(a, b):
#     return np.sqrt(((a-b)**2).sum(axis=-1)).mean()

# def kabsch_rmsd(new_coords, coords):
#     out = new_coords.T
#     target = coords.T
#     ret_R, ret_t = rigid_transform_3D(out, target, correct_reflection=False)
#     out = (ret_R@out) + ret_t
#     return compute_rmsd_np(target.T, out.T)


def compute_RMSD(a, b):
    # correct rmsd calculation.
    return np.sqrt((((a-b)**2).sum(axis=-1)).mean())

def kabsch_RMSD(new_coords, coords, return_rotran=False):
    out = new_coords.T
    target = coords.T
    ret_R, ret_t = rigid_transform_3D(out, target, correct_reflection=False)
    out = (ret_R@out) + ret_t
    if return_rotran:
        return compute_RMSD(target.T, out.T), ret_R, ret_t
    else:
        return compute_RMSD(target.T, out.T)



def download_alphaFold(uid_list, af_protein_path):
    for i, uid in enumerate(uid_list):
        if i % 10 == 0:
            print(i)
        if os.path.exists(f"{af_protein_path}/AF-{uid}-F1-model_v3.pdb"):
            continue
        out1 = os.system(f"wget -P {af_protein_path}/ https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v3.pdb")
        if out1 != 0:
            print(uid, out1)

def download_alphaFold_v2(uid_list, af_protein_path="/mnt/nas/research-data/luwei/downloaded_alphafold_structures/", version="4"):
    # v2 could also specific the version of alphafold structure.
    # af_protein_path
    downloaded_list = []
    for i, uid in enumerate(tqdm(uid_list)):
        # if i % 10 == 0:
        #     print(i)
        if os.path.exists(f"{af_protein_path}/AF-{uid}-F1-model_v{version}.pdb"):
            downloaded_list.append(uid)
            continue
        out1 = os.system(f"wget -P {af_protein_path}/ https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{version}.pdb")
        if out1 != 0:
            print(uid, out1)
        pFile = f"{af_protein_path}/AF-{uid}-F1-model_v{version}.pdb"
        if os.path.exists(pFile) and os.stat(pFile).st_size == 0:
            os.system(f"rm {pFile}")
            print(f"{pFile} is empty, removed")
        elif os.path.exists(pFile):
            downloaded_list.append(uid)
    return downloaded_list

def get_res_unique_id(residue):
    pdb, _, chain, (_, resid, insertion) = residue.full_id
    unique_id = f"{chain}_{resid}_{insertion}"
    return unique_id


def save_subset_protein(s, clean_res_list, proteinFile):
    res_id_list = set([get_res_unique_id(residue) for residue in clean_res_list])
    if proteinFile[-4:] == ".cif":
        io = MMCIFIO()
    else:
        io = PDBIO()

    class MySelect(Select):
        def accept_residue(self, residue, res_id_list=res_id_list):
            if get_res_unique_id(residue) in res_id_list:
                return True
            else:
                return False

    io.set_structure(s)
    io.save(proteinFile, MySelect())
    return None

def remove_hetero_v3(res_list, three_to_one=three_to_one, verbose=True, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print("not regular resname ", res.resname)
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero, removed")
    return clean_res_list

def filter_discontinous_region(idx_list):
    filter_list = []
    pre = idx_list[0]
    start = pre
    count = 0
    for i in idx_list[1:]:
        if pre + 1 != i:
            # print(count, i, start, pre)
            if count <= 3:
                # print(count, i, start, pre)
                for idx in range(start, pre+1):
                    filter_list.append(idx)
            count = 0
            start = i
        count += 1
        pre = i
    return filter_list

# def align_to_ref_pdb(ref_pdbFile = "/gxr/luwei/nustar/MASP2/clusters/1q3x_A.pdb",
#                      pdbFile = "/gxr/luwei/nustar/MASP2/af2_structures/AF-O00187-F1-model_v4.pdb",
#                      aligned_pdbFile = "/gxr/luwei/nustar/MASP2/af2_structures/O00187_aligned.pdb",
#                      keep_refFile="/gxr/luwei/nustar/MASP2/af2_structures/af2_1a3x.pdb",
#                      ref_chain=None, chain=None, ref_bfactor_cutoff=None, pdb_bfactor_cutoff=40, resultFile=None, verbose=False, 
#                      choose_pocket=None):
#     super_imposer = Superimposer()
#     parser = PDBParser(QUIET=True)
#     ref_pdb = parser.get_structure("x", ref_pdbFile)
#     if ref_chain:
#         ref_pdb = ref_pdb[0][ref_chain]

#     ref_all_res = remove_hetero_v3(ref_pdb.get_residues(), verbose=verbose, ensure_ca_exist=True, bfactor_cutoff=ref_bfactor_cutoff)
#     ref_seq = "".join([three_to_one.get(res.resname) for res in ref_all_res])

#     pdb = parser.get_structure("pdb", pdbFile)
#     if chain:
#         pdb = pdb[0][chain]
#     pdb_all_res = remove_hetero_v3(pdb.get_residues(), verbose=verbose, ensure_ca_exist=True, bfactor_cutoff=pdb_bfactor_cutoff)
#     pdb_seq = "".join([three_to_one.get(res.resname) for res in pdb_all_res])

#     result = align_to_original(ref_seq, pdb_seq)
#     # remove all dashes in ref_seq
#     info = get_aligned_index_info(result, columns=['ref_seq', 'ref_idx', 'pdb_seq', 'pdb_idx'])
#     info = info.query("ref_seq == pdb_seq").reset_index(drop=True)
#     # ensure ref_idx is continuous for at least 10 amino acids.
#     idx_list = info.ref_idx.values
#     filter_list = filter_discontinous_region(idx_list)
#     info = info.query("ref_idx not in @filter_list").reset_index(drop=True)
#     identity_ratio = (len(info) / (min(len(ref_seq), len(pdb_seq))))
#     # print(identity_ratio)
#     ref_ca_list = get_all_ca(ref_all_res)
#     pdb_ca_list = get_all_ca(pdb_all_res)

#     chosen_ref_ca_list = [ca for i, ca in enumerate(ref_ca_list) if i in info.ref_idx.values]
#     chosen_pdb_ca_list = [ca for i, ca in enumerate(pdb_ca_list) if i in info.pdb_idx.values]

#     keep_ref_res_list = [res for i, res in enumerate(ref_all_res) if i in info.ref_idx.values]
#     keep_pdb_res_list = [res for i, res in enumerate(pdb_all_res) if i in info.pdb_idx.values]

#     save_subset_protein(ref_pdb, keep_ref_res_list, keep_refFile)
#     if choose_pocket is not None:
#         pdb_ca_coords = np.stack([atom.coord for atom in chosen_pdb_ca_list])
#         mask = np.sqrt(((pdb_ca_coords - choose_pocket)**2).sum(axis=-1)) < 15
#         super_imposer.set_atoms(np.array(chosen_ref_ca_list)[mask], np.array(chosen_pdb_ca_list)[mask])
#         super_imposer.apply(pdb.get_atoms())
#     else:
#         super_imposer.set_atoms(chosen_ref_ca_list, chosen_pdb_ca_list)
#         super_imposer.apply(pdb.get_atoms())
#     save_subset_protein(pdb, keep_pdb_res_list, aligned_pdbFile)
#     if resultFile:
#         result = {
#                 'info':info,
#                  'ref_pdbFile':ref_pdbFile,
#                  'pdbFile':pdbFile,
#                  'aligned_pdbFile':aligned_pdbFile, 
#                  'keep_refFile':keep_refFile,
#                  'identity_ratio':identity_ratio, 'len_info':len(info), 
#                   'len_pdb_seq':len(pdb_seq), 'len_ref_seq':len(ref_seq),
#                 'rotrain':super_imposer.rotran, 'rmsd':super_imposer.rms}
#         np.save(resultFile, result)


def gap_mask_all_res(all_res, gap_mask):
    if gap_mask is None:
        return all_res
    assert len(all_res) == len(gap_mask)
    all_res = [res for res, isgap in zip(all_res, gap_mask) if isgap == '0']
    return all_res


def find_first_non_match_letter(s):
    for idx, l in enumerate(s):
        if l != '-':
            return idx
def find_last_non_match_letter(s):
    n = len(s)
    for idx in range(n-1, -1, -1):
        if s[idx] != '-':
            return idx+1

def remove_dash_in_ref_seq(result):
    # remove '-' in ref seq.
    new_result = [(x, y) for x, y in zip(*result) if x != '-']
    new_result = ["".join([x for x, y in new_result]),
                  "".join([y for x, y in new_result])]
    return new_result

def find_tail_idx(seq):
    i = 0
    for xi in seq:
        if xi == '-':
            i += 1
        else:
            return i
    return i

def transplant_residues(res, shift):
    for atom in res.get_atoms():
        atom.set_coord(atom.coord + shift)

def align_to_ref_pdb_v2(ref_pdbFile = "/gxr/luwei/nustar/MASP2/clusters/1q3x_A.pdb",
                     pdbFile = "/gxr/luwei/nustar/MASP2/af2_structures/AF-O00187-F1-model_v4.pdb",
                     aligned_pdbFile = "/gxr/luwei/nustar/MASP2/af2_structures/O00187_aligned.pdb",
                     keep_refFile="/gxr/luwei/nustar/MASP2/af2_structures/af2_1a3x.pdb",
                     ref_chain=None, chain=None, ref_bfactor_cutoff=None, pdb_bfactor_cutoff=40, resultFile=None, verbose=False, 
                     choose_pocket=None):
    super_imposer = Superimposer()
    parser = PDBParser(QUIET=True)
    ref_pdb = parser.get_structure("x", ref_pdbFile)
    if ref_chain:
        ref_pdb = ref_pdb[0][ref_chain]
    ref_all_res = remove_hetero_v3(ref_pdb.get_residues(), verbose=verbose, ensure_ca_exist=True, bfactor_cutoff=ref_bfactor_cutoff)
    ref_seq = "".join([three_to_one.get(res.resname) for res in ref_all_res])

    pdb = parser.get_structure("pdb", pdbFile)
    if chain:
        pdb = pdb[0][chain]
    else:
        chain = list(pdb[0].get_chains())[0].id
        pdb = pdb[0][chain]
    pdb_all_res = remove_hetero_v3(pdb.get_residues(), verbose=verbose, ensure_ca_exist=True, bfactor_cutoff=pdb_bfactor_cutoff)
    pdb_seq = "".join([three_to_one.get(res.resname) for res in pdb_all_res])

    raw_result = align_to_original(ref_seq, pdb_seq)

    # result = remove_dash_in_ref_seq(raw_result)
    # remove all dashes in ref_seq
    remove_pdb_idx_list = []
    r_0_list = []
    r_1_list = []
    ref_idx = pdb_idx = 0
    info = []
    for a,b in zip(*raw_result):
        if a == '-' and  b != '-':
            remove_pdb_idx_list.append(pdb_idx)
            pdb_idx += 1
            continue
        elif b == '-' and a != '-':
            ref_idx += 1
        elif a != '-' and b != '-':
            ref_idx += 1
            pdb_idx += 1
        else:
            print("error?")
            break
        r_0_list.append(a)
        r_1_list.append(b)
    result = ["".join(r_0_list), "".join(r_1_list)]
    pdb_all_res = [res for idx, res in enumerate(pdb_all_res) if idx not in remove_pdb_idx_list]

    start_idx, end_idx = find_first_non_match_letter(result[1]), find_last_non_match_letter(result[1])
    info = get_aligned_index_info(result, columns=['ref_seq', 'ref_idx', 'pdb_seq', 'pdb_idx'])
    # info = info.query("ref_seq == pdb_seq").reset_index(drop=True)

    # ensure ref_idx is continuous for at least 10 amino acids.
    idx_list = info.ref_idx.values
    filter_list = filter_discontinous_region(idx_list)
    info = info.query("ref_idx not in @filter_list").reset_index(drop=True)


    identity_ratio = (len(info.query("ref_seq==pdb_seq")) / (min(len(ref_seq), len(pdb_seq))))
    # print(identity_ratio)
    ref_ca_list = get_all_ca(ref_all_res)
    pdb_ca_list = get_all_ca(pdb_all_res)

    chosen_ref_ca_list = [ca for i, ca in enumerate(ref_ca_list) if i in info.ref_idx.values]
    chosen_pdb_ca_list = [ca for i, ca in enumerate(pdb_ca_list) if i in info.pdb_idx.values]

    # keep_ref_res_list = [res for i, res in enumerate(ref_all_res) if i in info.ref_idx.values]
    # keep_pdb_res_list = [res for i, res in enumerate(pdb_all_res) if i in info.pdb_idx.values]

    ref_all_res_removed_two_ends = ref_all_res[start_idx:end_idx]
    save_subset_protein(ref_pdb, ref_all_res_removed_two_ends, keep_refFile)

    if choose_pocket is not None:
        pdb_ca_coords = np.stack([atom.coord for atom in chosen_pdb_ca_list])
        mask = np.sqrt(((pdb_ca_coords - choose_pocket)**2).sum(axis=-1)) < 15
        if mask.sum() == 0:
            if verbose:
                print("no matched residue in contact with the ligand.")
            return None
        super_imposer.set_atoms(np.array(chosen_ref_ca_list)[mask], np.array(chosen_pdb_ca_list)[mask])
        super_imposer.apply(pdb.get_atoms())
    else:
        super_imposer.set_atoms(chosen_ref_ca_list, chosen_pdb_ca_list)
        super_imposer.apply(pdb.get_atoms())
        
    # save_subset_protein(pdb, keep_pdb_res_list, aligned_pdbFile)

    aligned_pdb_seq = result[1][start_idx:end_idx]
    # unmatched_res_all = [res for x, res in zip(aligned_pdb_seq, ref_all_res_removed_two_ends) if x=='-']

    pdb_idx = 0
    constructed_res_list = []
    for ref_idx, (x, ref_res) in enumerate(zip(aligned_pdb_seq, ref_all_res_removed_two_ends)):
        if x == '-':
            # transplanted residues will be moved to connect two ends.
            if new_unmatched:
                head_idx = ref_idx-1
                head_ref = ref_all_res_removed_two_ends[head_idx]
                head_pdb = pdb_all_res[pdb_idx-1]
                new_unmatched = False
                unmatched_width = find_tail_idx(aligned_pdb_seq[ref_idx:])
                tail_idx = ref_idx + unmatched_width
                tail_ref = ref_all_res_removed_two_ends[tail_idx]
                tail_pdb = pdb_all_res[pdb_idx]
                head_shift = head_pdb['CA'].coord - head_ref['CA'].coord
                tail_shift = tail_pdb['CA'].coord - tail_ref['CA'].coord
            
            decay = 1 - ((ref_idx - head_idx) / (unmatched_width + 1))
            shift = (head_shift * decay) + (tail_shift * (1 - decay))
            transplant_residues(ref_res, shift)
            ref_res.detach_parent()
            ref_res.id = (' ', ref_idx+1, ' ')
            constructed_res_list.append(ref_res)
        else:
            pdb_all_res[pdb_idx].detach_parent()
            pdb_all_res[pdb_idx].id = (' ', ref_idx+1, ' ')
            constructed_res_list.append(pdb_all_res[pdb_idx])
            pdb_idx += 1
            new_unmatched = True
            
    pdb.child_list = constructed_res_list

    io = PDBIO()
    io.set_structure(pdb)
    io.save(aligned_pdbFile)
    if resultFile:
        result = {
                'info':info,
                 'ref_pdbFile':ref_pdbFile,
                 'pdbFile':pdbFile,
                 'aligned_pdbFile':aligned_pdbFile, 
                 'keep_refFile':keep_refFile,
                 'identity_ratio':identity_ratio, 'len_info':len(info), 
                'len_pdb_seq':len(pdb_seq), 'len_ref_seq':len(ref_seq), 
                'start_idx':start_idx, 'end_idx':end_idx,
                'result_seq_ref':result[0], 'result_seq_pdb':result[1],
                'raw_result_ref':raw_result[0], 'raw_result_pdb':raw_result[1],
                'rotrain':super_imposer.rotran, 'rmsd':super_imposer.rms}
        np.save(resultFile, result)



def locate_starting_site(raw_result_ref, raw_result_pdb):
    # start when consecutively matched 10 residues.
    match_count = 0
    for idx, (ref_res, pdb_res) in enumerate(zip(raw_result_ref, raw_result_pdb)):
        if ref_res == pdb_res:
            match_count += 1
        else:
            match_count = 0
        if match_count == 10:
            return idx - 9
    return -1

def locate_ending_site(raw_result_ref, raw_result_pdb):
    # end when consecutively matched 10 residues.
    match_count = 0
    n = len(raw_result_ref)
    # for raw_idx, (ref_res, ref_pdb) in enumerate(zip(raw_result_ref[::-1], raw_result_pdb[::-1])):
    for idx in range(n-1, -1, -1):
        ref_res = raw_result_ref[idx]
        pdb_res = raw_result_pdb[idx]
        if ref_res == pdb_res:
            match_count += 1
        else:
            match_count = 0
        if match_count == 10:
            return idx + 10
    return -1

def get_chosen_ca_list(raw_result_ref, raw_result_pdb, start_idx, end_idx, ref_ca_list, pdb_ca_list):
    chosen_ref_ca_list = []
    chosen_pdb_ca_list = []
    ref_idx = pdb_idx = 0
    for idx, (ref_res, pdb_res) in enumerate(zip(raw_result_ref, raw_result_pdb)):
        if ref_res == '-' and  pdb_res != '-':
            pdb_idx += 1
            continue
        elif pdb_res == '-' and ref_res != '-':
            ref_idx += 1
        elif ref_res != '-' and pdb_res != '-':
            if idx >= start_idx and idx < end_idx:
                chosen_ref_ca_list.append(ref_ca_list[ref_idx])
                chosen_pdb_ca_list.append(pdb_ca_list[pdb_idx])
            ref_idx += 1
            pdb_idx += 1
        else:
            print("error?")
            break
    return chosen_ref_ca_list, chosen_pdb_ca_list

def get_ref_all_res_removed_two_ends(raw_result_ref, raw_result_pdb, start_idx, end_idx, ref_all_res):
    ref_all_res_removed_two_ends = []
    ref_idx = 0
    for idx, (ref_res, pdb_res) in enumerate(zip(raw_result_ref, raw_result_pdb)):
        if ref_res == '-' and  pdb_res != '-':
            continue
        elif pdb_res == '-' and ref_res != '-':
            if idx >= start_idx and idx < end_idx:
                ref_all_res_removed_two_ends.append(ref_all_res[ref_idx])
            ref_idx += 1
        elif ref_res != '-' and pdb_res != '-':
            if idx >= start_idx and idx < end_idx:
                ref_all_res_removed_two_ends.append(ref_all_res[ref_idx])
            ref_idx += 1
        else:
            print("error?")
            break
    return ref_all_res_removed_two_ends


def get_constructed_res_list(raw_result_ref, raw_result_pdb, start_idx, end_idx, ref_all_res, pdb_all_res):
    idx = ref_idx = pdb_idx = 0
    constructed_res_list = []
    for i, (ref_res, pdb_res) in enumerate(zip(raw_result_ref, raw_result_pdb)):
        if ref_res == '-' and  pdb_res != '-':
            pdb_idx += 1
        elif pdb_res == '-' and ref_res != '-':
            if i >= start_idx and i < end_idx:
                # transplanted residues will be moved to connect two ends.
                if new_unmatched:
                    head_idx = ref_idx-1
                    head_ref = ref_all_res[head_idx]
                    head_pdb = pdb_all_res[pdb_idx-1]
                    new_unmatched = False
                    unmatched_width = find_tail_idx(raw_result_pdb[i:])
                    tail_idx = ref_idx + unmatched_width
                    tail_ref = ref_all_res[tail_idx]
                    tail_pdb = pdb_all_res[pdb_idx]
                    head_shift = head_pdb['CA'].coord - head_ref['CA'].coord
                    tail_shift = tail_pdb['CA'].coord - tail_ref['CA'].coord

                decay = 1 - ((ref_idx - head_idx) / (unmatched_width + 1))
                shift = (head_shift * decay) + (tail_shift * (1 - decay))
                ref_residue = ref_all_res[ref_idx]
                transplant_residues(ref_residue, shift)
                ref_residue.detach_parent()
                ref_residue.id = (' ', idx+1, ' ')
                constructed_res_list.append(ref_residue)
            idx += 1
            ref_idx += 1

        elif ref_res != '-' and pdb_res != '-':
            if i >= start_idx and i < end_idx:
                pdb_all_res[pdb_idx].detach_parent()
                pdb_all_res[pdb_idx].id = (' ', idx+1, ' ')
                constructed_res_list.append(pdb_all_res[pdb_idx])
                new_unmatched = True
            idx += 1
            ref_idx += 1
            pdb_idx += 1
        else:
            print("error?")
            break
    return constructed_res_list

def align_to_ref_pdb_v3(
                     ref_pdbFile="/gxr/luwei/nustar/MASP2/af2_structures/AF-O00187-F1-model_v4.pdb",
                     pdbFile="/gxr/luwei/nustar/MASP2/clusters/1q3x_A.pdb",
                     aligned_pdbFile="/gxr/luwei/nustar/MASP2/af2_structures/O00187_aligned.pdb",
                     keep_refFile="/gxr/luwei/nustar/MASP2/af2_structures/af2_1a3x.pdb",
                     ref_chain=None, chain=None, ref_bfactor_cutoff=None, pdb_bfactor_cutoff=None, resultFile=None, verbose=False, 
                     choose_pocket=None, ref_is_af2=True):
    super_imposer = Superimposer()
    parser = MMCIFParser(QUIET=True) if ref_pdbFile[-4:] == ".cif" else PDBParser(QUIET=True)
    ref_pdb = parser.get_structure("x", ref_pdbFile)
    if ref_chain:
        ref_pdb = ref_pdb[0][ref_chain]
    ref_all_res = remove_hetero_v3(ref_pdb.get_residues(), verbose=verbose, ensure_ca_exist=True, bfactor_cutoff=ref_bfactor_cutoff)
    ref_seq = "".join([three_to_one.get(res.resname) for res in ref_all_res])

    parser = MMCIFParser(QUIET=True) if pdbFile[-4:] == ".cif" else PDBParser(QUIET=True)
    pdb = parser.get_structure("pdb", pdbFile)
    if chain:
        pdb = pdb[0][chain]
    else:
        if 0 not in pdb:
            if verbose:
                print("pdb file is probably empty")
            return None
        chain = list(pdb[0].get_chains())[0].id
        pdb = pdb[0][chain]

    pdb_all_res = remove_hetero_v3(pdb.get_residues(), verbose=verbose, ensure_ca_exist=True, bfactor_cutoff=pdb_bfactor_cutoff)
    pdb_seq = "".join([three_to_one.get(res.resname) for res in pdb_all_res])

    raw_result = align_to_original(ref_seq, pdb_seq)
    if len(raw_result) == 0:
        if verbose:
            print("could do the alignment, probably because one sequence is too short")
        return None
    raw_result_ref, raw_result_pdb = raw_result[0], raw_result[1]
    # remove unmatched seq at two ends first.
    start_idx = locate_starting_site(raw_result_ref, raw_result_pdb)
    end_idx = locate_ending_site(raw_result_ref, raw_result_pdb)
    # raw_result_ref[start_idx:end_idx]

    # print(raw_result_ref[start_idx:end_idx])
    # print(raw_result_pdb[start_idx:end_idx])

    ref_seq = raw_result_ref[start_idx:end_idx]
    ref_gap_count = ref_seq.count("-")

    if ref_gap_count > 0:
        if ref_is_af2:
            if verbose:
                print("ref pdb has gap, exit.")
            return None
        else:
            if verbose:
                print("ref pdb has gap, but its ok, I guess.")

    if start_idx == -1 or end_idx == -1:
        if verbose:
            print("could not align the sequence.")
        return None
    # remove gap in ref_seq.
    ref_ca_list = get_all_ca(ref_all_res)
    pdb_ca_list = get_all_ca(pdb_all_res)

    chosen_ref_ca_list, chosen_pdb_ca_list = get_chosen_ca_list(raw_result_ref, raw_result_pdb, start_idx, end_idx, ref_ca_list, pdb_ca_list)

    ref_all_res_removed_two_ends = get_ref_all_res_removed_two_ends(raw_result_ref, raw_result_pdb, start_idx, end_idx, ref_all_res)
    save_subset_protein(ref_pdb, ref_all_res_removed_two_ends, keep_refFile)

    if choose_pocket is not None:
        pdb_ca_coords = np.stack([atom.coord for atom in chosen_pdb_ca_list])
        mask = np.sqrt(((pdb_ca_coords - choose_pocket)**2).sum(axis=-1)) < 15
        if mask.sum() == 0:
            if verbose:
                print("no matched residue in contact with the ligand.")
            return None
        super_imposer.set_atoms(np.array(chosen_ref_ca_list)[mask], np.array(chosen_pdb_ca_list)[mask])
        super_imposer.apply(pdb.get_atoms())
    else:
        super_imposer.set_atoms(chosen_ref_ca_list, chosen_pdb_ca_list)
        super_imposer.apply(pdb.get_atoms())


    constructed_res_list = get_constructed_res_list(raw_result_ref, raw_result_pdb, start_idx, end_idx, ref_all_res, pdb_all_res)
    pdb.child_list = constructed_res_list
    if aligned_pdbFile[-4:] == ".cif":
        io = MMCIFIO()
    else:
        io = PDBIO()
    io.set_structure(pdb)
    # class NotDisordered(Select):
    #     def accept_atom(self, atom):
    #         return not atom.is_disordered() or atom.get_altloc() == "A"
    # io.save(aligned_pdbFile, select=NotDisordered())
    class NotDisordered_NoHydrogen(Select):
        def accept_atom(self, atom):
            if atom.element == 'H' or atom.element == 'D':
                return False
            return not atom.is_disordered() or atom.get_altloc() == "A"
    io.save(aligned_pdbFile, select=NotDisordered_NoHydrogen())

    identity_ratio = (np.array(list(raw_result_ref[start_idx:end_idx])) == np.array(list(raw_result_pdb[start_idx:end_idx]))).sum() / (end_idx - start_idx)
    if resultFile:
        result = {
                # 'info':info,
                 'ref_pdbFile':ref_pdbFile,
                 'pdbFile':pdbFile,
                 'aligned_pdbFile':aligned_pdbFile, 
                 'keep_refFile':keep_refFile,
                 'identity_ratio':identity_ratio, 'len_info':(end_idx - start_idx), # 'len_info':len(info), 
                'len_pdb_seq':len(pdb_seq), 'len_ref_seq':len(ref_seq), 
                'start_idx':start_idx, 'end_idx':end_idx,
                'result_seq_ref':raw_result_ref[start_idx:end_idx], 
                'result_seq_pdb':raw_result_pdb[start_idx:end_idx],
                'raw_result_ref':raw_result[0], 'raw_result_pdb':raw_result[1],
                'rotrain':super_imposer.rotran, 'rmsd':super_imposer.rms}
        np.save(resultFile, result)

def align_sdf_based_on_resultFile(compoundName, sdf_fileName, ligand_folder, resultFile):
    to_sdf_fileName = f'{ligand_folder}/{compoundName}.sdf'
    rot, trans = resultFile['rotrain'][0], resultFile['rotrain'][1]
    align_sdf_based_on_resultFile_v2(sdf_fileName, to_sdf_fileName, rot, trans)


def align_sdf_based_on_resultFile_v2(sdf_fileName, to_sdf_fileName, rot, trans):
    mol = Chem.MolFromMolFile(sdf_fileName)
    coords = mol.GetConformer().GetPositions()

    new_coords = np.dot(coords, rot) + trans
    # new_coords = np.dot(coords - a['rotrain'][1], a['rotrain'][0].T)
    # f'{ligand_folder}/{compoundName}.sdf'
    w = Chem.SDWriter(to_sdf_fileName)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()


def align_protein_to_ground_truth(input_protein_file, input_ligand_file, 
                                 pred_ligand_file, pred_protein_file,
                                 to_ligand_file, to_protein_file, gap_mask=None):
    mol = Chem.MolFromMolFile(input_ligand_file)
    choose_pocket = mol.GetConformer().GetPositions().mean(axis=0).reshape(1, 3)

    super_imposer = Superimposer()
    parser = MMCIFParser(QUIET=True) if input_protein_file[-4:] == ".cif" else PDBParser(QUIET=True)
    ref_pdb = parser.get_structure("x", input_protein_file)
    # ref_all_res = list(ref_pdb.get_residues())
    ref_all_res = gap_mask_all_res(list(ref_pdb.get_residues()), gap_mask)
    ref_ca_list = [res['CA'] for res in ref_all_res]
    ref_pdb_ca_coords = np.stack([atom.coord for atom in ref_ca_list])

    parser = MMCIFParser(QUIET=True) if pred_protein_file[-4:] == ".cif" else PDBParser(QUIET=True)
    pdb = parser.get_structure("pdb", pred_protein_file)
    # all_res = list(pdb.get_residues())
    all_res = gap_mask_all_res(list(pdb.get_residues()), gap_mask)
    ca_list = [res['CA'] for res in all_res]
    pdb_ca_coords = np.stack([atom.coord for atom in ca_list])

    mask = np.sqrt(((ref_pdb_ca_coords - choose_pocket)**2).sum(axis=-1)) < 15
    if mask.sum() == 0:
        if verbose:
            print("no matched residue in contact with the ligand.")
        # return None
    super_imposer.set_atoms(np.array(ref_ca_list)[mask], np.array(ca_list)[mask])
    super_imposer.apply(pdb.get_atoms())

    io = PDBIO()
    io.set_structure(pdb)
    io.save(to_protein_file)

    rot, trans = super_imposer.rotran
    align_sdf_based_on_resultFile_v2(pred_ligand_file, to_ligand_file, rot, trans)


import requests as rq
def get_sdf(x):
    try:
        pdb_id, ligand_id, chain, toFile = x
        url = f'https://models.rcsb.org/v1/{pdb_id}/ligand?auth_comp_id={ligand_id}&auth_asym_id={chain}&encoding=sdf&filename=ligand.sdf'
        r = rq.get(url)
        # toFolder = "/mnt/nas/research-data/luwei/biolip/sdfs/"
        open(toFile,'wb').write(r.content)
    except Exception as e:
        print(e)
        print(x)
    return

def fix_sdf_atom_count(original_sdfFile, new_sdfFile):
    # sdf may contain missing hydrogen. We want to fix the atom count in sdf file.
    with open(original_sdfFile) as f:
        out = f.readlines()
    atom_count = 0
    bond_count = 0
    read_atom = True
    read_bond = True
    for line in out[4:]:
        if read_atom and len(line.split()) == 16:
            atom_count += 1
            continue
        read_atom = False
        if read_bond and len(line.split()) == 7:
            bond_count += 1
            continue
        read_bond = False
    o = " ".join([str(atom_count), str(bond_count), '0', '0', '0', '0', '0', '0', '0', '0', '0'])
    result = out[:3] + [o+"\n"] + out[4:]
    with open(new_sdfFile, "w") as f:
        f.write("".join(result))


def write_mol_to_sdf(mol, fileName):
    w = Chem.SDWriter(fileName)
    w.write(mol)
    w.close()

def write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = new_coords[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()


# mols = [Chem.MolFromSmiles(s) for s in t.sample(30)['smiles'].values]
# Chem.Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250,250),useSVG=True)
# mols = [Chem.MolFromSmiles(s) for s in t.sample(30)['smiles'].values]
# Chem.Draw.MolsToGridImage(mols, molsPerRow=5, legends=list(t1['cid'].values))
# mols = [Chem.MolFromSmiles(s) for s in smiles_list]
# Chem.Draw.MolsToGridImage(mols, molsPerRow=5, legends=ligandName_list)

def max_length_subsmiles(smiles):
    try:
        return max(smiles.split('.'), key=len, default='')
    except:
        return ""

def get_canonical_smiles(smiles):
    try:
        # smiles = smiles.split(".")[-1]
        smiles = max_length_subsmiles(smiles)
        can_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        can_smiles = ""
    return can_smiles


from Bio.PDB import cealign
def cealign_python(refFile, pdbFile, toFile):
    parser = PDBParser(QUIET=True)
    ref = parser.get_structure("x", refFile)
    pred = parser.get_structure("x", pdbFile)
    ce = cealign.CEAligner()
    ce.set_reference(ref)
    ce.align(pred)
    pdbio = PDBIO()
    pdbio.set_structure(pred)
    pdbio.save(toFile)
    return ce.rms


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


import warnings
def request_limited(url: str,
                    rtype: str = "GET",
                    num_attempts: int = 3,
                    sleep_time=0.5,
                    **kwargs) -> Optional[requests.models.Response]:
    """
    HTML request with rate-limiting base on response code
    Parameters
    ----------
    url : str
        The url for the request
    rtype : str
        The request type (oneof ["GET", "POST"])
    num_attempts : int
        In case of a failed retrieval, the number of attempts to try again
    sleep_time : int
        The amount of time to wait between requests, in case of
        API rate limits
    **kwargs : dict
        The keyword arguments to pass to the request
    Returns
    -------
    response : requests.models.Response
        The server response object. Only returned if request was successful,
        otherwise returns None.
    """

    if rtype not in ["GET", "POST"]:
        warnings.warn("Request type not recognized")
        return None

    total_attempts = 0
    while (total_attempts <= num_attempts):
        if rtype == "GET":
            response = requests.get(url, **kwargs)
        elif rtype == "POST":
            response = requests.post(url, **kwargs)

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            curr_sleep = (1 + total_attempts) * sleep_time
            warnings.warn("Too many requests, waiting " + str(curr_sleep) +
                          " s")
            time.sleep(curr_sleep)
        elif 500 <= response.status_code < 600:
            warnings.warn("Server error encountered. Retrying")
        total_attempts += 1

    warnings.warn("Too many failures on requests. Exiting...")
    return None

# def get_pdb_info(pdb_id, url_root='https://data.rcsb.org/rest/v1/core/entry/'):
#     '''Look up all information about a given PDB ID
#     Parameters
#     ----------
#     pdb_id : string
#         A 4 character string giving a pdb entry of interest
#     url_root : string
#         The string root of the specific url for the request type
#     Returns
#     -------
#     out : dict()
#         An ordered dictionary object corresponding to entry information
#     '''
#     pdb_id = pdb_id.replace(":", "/")  # replace old entry identifier
#     url = url_root + pdb_id
#     response = request_limited(url)

#     if response is None or response.status_code != 200:
#         warnings.warn("Retrieval failed, returning None")
#         return None

#     result = str(response.text)

#     out = json.loads(result)

#     return out

def get_ligand_name(c, verbose=True, n_atoms_thr=10):
    regular_res_list = []
    hetero_res_list = []
    n_atom_list = []
    for res in c.get_residues():
        if res.id[0] == " " or res.id[0] == 'H_MSE':
            regular_res_list.append(res)
        else:
            if res.id[0] == "W":
                # skip water.
                continue
            n_atoms = len(list(res.get_atoms()))
            if n_atoms < n_atoms_thr:
                if verbose:
                    print("Skipping ", res.resname, "having ", n_atoms, "atoms, less than 15")
                continue
            hetero_res_list.append(res)
            n_atom_list.append(n_atoms)
    if len(hetero_res_list) > 1:
        return ",".join([f"{res.resname}_{n_atom}" for res, n_atom in zip(hetero_res_list, n_atom_list)])
    return ",".join([res.resname for res in hetero_res_list])


def save_clean_protein(s, keep_chain, toFile):
    class MySelect(Select):
        def accept_residue(self, residue, keep_chain=keep_chain):
            pdb, _, chain, (hetero, resid, insertion) = residue.full_id
            if chain == keep_chain:
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
# moved to analysis

# def get_atom_unique_id(atom):
#     pdb, _, chain, (_, resid, insertion), (atomName, altloc) = atom.full_id
#     unique_id = f"{atomName}_{altloc}"
#     return unique_id

# def get_mapped_atom_coords(ref_s, s):
#     ref_all_res = list(ref_s.get_residues())
#     all_res = list(s.get_residues())

#     ref_atom_coord_list = []
#     atom_coord_list = []
#     skip_atom_count = 0
#     for ref_res, res in zip(ref_all_res, all_res):
#         for ref_atom in ref_res:
#             if ref_atom.id in res:
#                 atom = res[ref_atom.id]
#                 ref_atom_coord_list.append(ref_atom.coord)
#                 atom_coord_list.append(atom.coord)
#             elif ref_atom.element == 'H' or ref_atom.element == 'D':
#                 # D is Deuterium
#                 pass
#             else:
#                 skip_atom_count += 1
#                 pass
#                 # print(ref_atom)
#     if skip_atom_count > 10:
#         print(ref_s.id, "skipped more than 10 atoms, sometime might be wrong.")
#     ref_atom_coords = np.array(ref_atom_coord_list)
#     atom_coords = np.array(atom_coord_list)
#     return ref_atom_coords, atom_coords

# from scipy.spatial.distance import cdist
# def compute_protein_RMSD(pdbFile, pred_proteinFile, ref_ligand_coords=None):
#     parser = PDBParser(QUIET=True)
#     ref_s = parser.get_structure(pdbFile, pdbFile)
#     ref_all_atoms = list(ref_s.get_atoms())

#     s = parser.get_structure(pred_proteinFile, pred_proteinFile)
#     all_atoms = list(s.get_atoms())

#     ref_atom_coords, atom_coords = get_mapped_atom_coords(ref_s, s)
#     if ref_ligand_coords is not None:
#         pocket_threshold = 5
#         pocket_mask = cdist(ref_atom_coords, ref_ligand_coords).min(axis=-1) < pocket_threshold
#         if pocket_mask.sum() == 0:
#             return compute_RMSD(ref_atom_coords, atom_coords), -1
#         pocket_rmsd = compute_RMSD(ref_atom_coords[pocket_mask], atom_coords[pocket_mask])
#         return compute_RMSD(ref_atom_coords, atom_coords), pocket_rmsd
#     else:
#         return compute_RMSD(ref_atom_coords, atom_coords)