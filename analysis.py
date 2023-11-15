import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import rdkit.Chem as Chem

pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 100)

import torch
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import Superimposer
from Bio.PDB import PDBIO, Select

from rdkit.Geometry import Point3D

import signal
from contextlib import contextmanager
from spyrmsd import rmsd as spyrmsd_rmsd
from spyrmsd import molecule as spyrmsd_molecule

# from utils import compute_RMSD, compute_clash_score, compute_side_chain_metrics

from utils.clash import compute_side_chain_metrics
from helper_functions import compute_RMSD, gap_mask_all_res, align_protein_to_ground_truth, align_sdf_based_on_resultFile_v2
from compute_lddt import compute_lddt   # in helper function folder.
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    with time_limit(5):
        mol = spyrmsd_molecule.Molecule.from_rdkit(mol)
        mol2 = spyrmsd_molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = spyrmsd_rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
        )
        return RMSD


def get_atom_unique_id(atom):
    pdb, _, chain, (_, resid, insertion), (atomName, altloc) = atom.full_id
    unique_id = f"{atomName}_{altloc}"
    return unique_id

def get_mapped_atom_coords(ref_all_res, all_res):
    ref_atom_coord_list = []
    atom_coord_list = []
    skip_atom_count = 0
    for ref_res, res in zip(ref_all_res, all_res):
        for ref_atom in ref_res:
            if ref_atom.id in res:
                atom = res[ref_atom.id]
                ref_atom_coord_list.append(ref_atom.coord)
                atom_coord_list.append(atom.coord)
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
    return ref_atom_coords, atom_coords

from scipy.spatial.distance import cdist
def compute_protein_RMSD(pdbFile, pred_proteinFile, ref_ligand_coords=None, gap_mask=None):
    if pdbFile[-4:] == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    ref_s = parser.get_structure(pdbFile, pdbFile)
    ref_all_res = gap_mask_all_res(list(ref_s.get_residues()), gap_mask)
    # ref_all_atoms = list(ref_s.get_atoms())
    if pred_proteinFile[-4:] == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    s = parser.get_structure(pred_proteinFile, pred_proteinFile)
    all_res = gap_mask_all_res(list(s.get_residues()), gap_mask)
    # all_atoms = list(s.get_atoms())

    ref_atom_coords, atom_coords = get_mapped_atom_coords(ref_all_res, all_res)
    if ref_ligand_coords is not None:
        pocket_threshold = 5
        pocket_mask = cdist(ref_atom_coords, ref_ligand_coords).min(axis=-1) < pocket_threshold
        if pocket_mask.sum() == 0:
            return compute_RMSD(ref_atom_coords, atom_coords), -1
        pocket_rmsd = compute_RMSD(ref_atom_coords[pocket_mask], atom_coords[pocket_mask])
        return compute_RMSD(ref_atom_coords, atom_coords), pocket_rmsd
    else:
        return compute_RMSD(ref_atom_coords, atom_coords)



# rmsd = get_symmetry_rmsd(ref_mol, ref_mol.GetConformer().GetPositions(), Chem.MolFromMolFile(l).GetConformer().GetPositions())
def get_header_and_data(runName):
    header, dName = runName.split("__")
#     if header != 'diffdock':
#         break

    if dName in ['test_af2', 'test_crystal', 'test_af2_v9_0419']:
        dataset_name = 'pdbbind_v9'
    elif dName in ['test_af2_v10', 'test_crystal_v10', 'test_af2_v10_0419']:
        dataset_name = 'pdbbind_v10'
    elif dName in ['test_af2_v11', 'test_crystal_v11']:
        dataset_name = 'pdbbind_v11'
    elif dName in ['test_pocketminer_v2']:
        dataset_name = 'pocketminer_v2'
    elif dName in ['test_pocketminer_apo']:
        dataset_name = 'pocketminer_apo'
    elif dName in ['test_kinase']:
        dataset_name = 'kinase'
    elif dName in ['test_highcite']:
        dataset_name = 'highcite'
    elif dName in ['test_imatinib']:
        dataset_name = 'imatinib'
    elif dName in ['test_unseen']:
        dataset_name = 'unseen'
    elif dName in ['test_new_binding_site']:
        dataset_name = 'new_binding_site'
    elif dName in ['test_kinase_has_clash']:
        dataset_name = 'kinase_has_clash'
    elif dName in ['test_major_family']:
        dataset_name = 'major_family'
    elif dName in ['test_major_drug']:
        dataset_name = 'major_drug_targets'
    else:
        return header, dName
        # print("skipping,", dName)
        # return -1, -1
    if dName in ['test_crystal', 'test_crystal_v10', 'test_crystal_v11']:
        header = header + "_crystal"
    if dName in ['test_af2_v10_0419', 'test_af2_v9_0419']:
        header = header + "_0419"
    return header, dataset_name



def move_and_align(x):
    if len(x) == 5:
        folder_loc, header, cur_pre, entryName, affinityFile = x
        gap_mask = None
    else:
        folder_loc, header, cur_pre, entryName, affinityFile, gap_mask = x
    affinity_info = []
    # if os.path.exists(affinityFile):
    #     return None
    for entry in os.listdir(folder_loc):
        # print(entry)
        if entry == "rank1.sdf":
            continue
        if "_relaxed." in entry:
            continue
        if "_step" in entry:
            continue

        if 'diffdock' in header:
            if entry[-4:] == '.sdf':
                if 'rank' not in entry:
                    continue
                # print(entry)
                rank, confidence = entry.split("_confidence")
                rank = int(rank[4:])
                confidence = float(confidence[:-4])
                os.system(f"cp {folder_loc}/{entry} {cur_pre}/{header}_rank{rank}_ligand.sdf")
        else:
            if entry[-4:] == '.sdf' and 'confidence' in entry:
                # entry = "rank28_ligand_lddt0.25_affinity2.57_confidence-3.44.sdf"
                rank, _, lddt, affinity, confidence = entry.split("_")
                rank = int(rank[4:])
                lddt = float(lddt[4:])
                affinity = float(affinity[8:])
                confidence = float(confidence[10:-4])
                os.system(f"cp {folder_loc}/{entry} {cur_pre}/{header}_rank{rank}_ligand_raw.sdf")
                affinity_info.append([header, entryName, rank, lddt, affinity, confidence])
            elif entry[-4:] == '.pdb' and 'confidence' in entry:
                # entry = "rank9_receptor_lddt0.24_affinity2.63_confidence-2.89.pdb"
                rank, _, lddt, affinity, confidence = entry.split("_")
                rank = int(rank[4:])
                lddt = float(lddt[4:])
                affinity = float(affinity[8:])
                confidence = float(confidence[10:-4])
                os.system(f"cp {folder_loc}/{entry} {cur_pre}/{header}_rank{rank}_receptor_raw.pdb")
            elif entry[-4:] == '.sdf' and 'lddt' in entry:
                # entry = "rank1_ligand_lddt0.54_affinity3.37.sdf"
                rank, _, lddt, affinity = entry.split("_")
                rank = int(rank[4:])
                lddt = float(lddt[4:])
                affinity = float(affinity[8:-4])
                os.system(f"cp {folder_loc}/{entry} {cur_pre}/{header}_rank{rank}_ligand_raw.sdf")
                affinity_info.append([header, entryName, rank, lddt, affinity, 0])
            elif entry[-4:] == '.pdb' and 'lddt' in entry:
                rank, _, lddt, affinity = entry.split("_")
                rank = int(rank[4:])
                lddt = float(lddt[4:])
                affinity = float(affinity[8:-4])
                os.system(f"cp {folder_loc}/{entry} {cur_pre}/{header}_rank{rank}_receptor_raw.pdb")
            else:
                # print(entry, "skipped")
                pass
    if 'diffdock' not in header:
        if 'refined' in header:
            rank_list = [1]
        else:
            rank_list = range(1, 41)

        input_protein_file = f"{cur_pre}/{entryName}_holo_protein.pdb"
        if not os.path.exists(input_protein_file):
            input_protein_file = f"{cur_pre}/{entryName}_holo_protein.cif"

        input_ligand_file = f"{cur_pre}/{entryName}_ligand.sdf"
        for rank in rank_list:
            pred_ligand_file = f"{cur_pre}/{header}_rank{rank}_ligand_raw.sdf"
            pred_protein_file = f"{cur_pre}/{header}_rank{rank}_receptor_raw.pdb"

            to_ligand_file = f"{cur_pre}/{header}_rank{rank}_ligand.sdf"
            to_protein_file = f"{cur_pre}/{header}_rank{rank}_receptor.pdb"
            try:
                align_protein_to_ground_truth(input_protein_file, input_ligand_file,
                             pred_ligand_file, pred_protein_file,
                             to_ligand_file, to_protein_file, gap_mask=gap_mask)
            except Exception as e:
                print("aligned error,", e)
                print(input_protein_file, input_ligand_file,
                             pred_ligand_file, pred_protein_file,
                             to_ligand_file, to_protein_file)
                pass
    affinity_info = pd.DataFrame(affinity_info, columns=['conformation', 'entryName', 'rank', 'lddt', 'affinity', 'confidence'])
    affinity_info.to_csv(affinityFile, index=0)
#     affinity_info = pd.DataFrame(affinity_info, columns=['runName', 'conformation', 'dataset_name', 'entryName', 'i', 'rank', 'lddt', 'affinity'])
#     affinity_info.to_csv(f"{pre}/affinity_{runName}.csv", index=0)


def compute_tmscore(modelFile, refFile):
    cmd = f"/home/luw/packages/USalign/USalign {modelFile} {refFile}"
    # cmd = f"/gxr/luwei/bin/USalign {modelFile} {refFile}"
    with time_limit(120):
        result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
        # result.stdout
        out = result.stdout.decode().split("\n")
        for line in out:
            if line[:9] == 'TM-score=':
                score = float(line[9:18])
                break
        result = {"tmscore":score, "output":out, }
        return result

def compute_clash_and_rmsd(x):
    if len(x) == 6:
        cur_pre, header, pdbFile, af2File, ligandFile, infoFile = x
        gap_mask = None
    else:
        cur_pre, header, pdbFile, af2File, ligandFile, infoFile, gap_mask = x
    # if os.path.exists(infoFile):
    #     return None
    # print(x)
    info = []
    clash_query_list = []
    rank_list = range(1, 41)
    for rank in rank_list:
        pred_proteinFile = f"{cur_pre}/{header}_rank{rank}_receptor.pdb"
        # if header in ['diffdock_af2_all', 'diffdock_random1', 'diffdock_small_af2_all']:
        if 'crystal' in header and 'diffdock' in header:
            pred_proteinFile = pdbFile
        elif "diffdock" in header:
            pred_proteinFile = af2File
        pred_ligandFile = f"{cur_pre}/{header}_rank{rank}_ligand.sdf"
        clash_query_list.append((f'{header}', rank, pred_proteinFile, pred_ligandFile))

    ref_mol = Chem.MolFromMolFile(ligandFile)
    ref_ligand_coords = ref_mol.GetConformer().GetPositions()

    for (conformation, rank, p, l) in clash_query_list:
        try:
            rmsd = compute_RMSD(Chem.MolFromMolFile(l).GetConformer().GetPositions(),
                            ref_ligand_coords)
            rmsd_sym = get_symmetry_rmsd(ref_mol, ref_ligand_coords, Chem.MolFromMolFile(l).GetConformer().GetPositions())
        except Exception as e:
            # print(e, l)
            if str(e) == "Timed out!":
                rmsd_sym = rmsd
            else:
                # skip_pdb_list.append(entryName)
                print(e, "compute RMSD")
                continue
        # info.append([conformation, rank, -1, -1, -1, -1, -1, -1, 0, 0, 0, rmsd_sym, rmsd])
        # continue
        try:
            clashScore, overlap, clash_n, n = compute_side_chain_metrics(p, l, verbose=True)
            overlap_mean = overlap.mean() if len(overlap) > 0 else 0
        except Exception as e:
            print("compute side chain metrics error", p, l)
            clashScore = -1.0
            overlap_mean = 0
            clash_n = n = 0
        # protein_rmsd = compute_protein_RMSD(pdbFile, p)
        try:
            with time_limit(60):
                protein_rmsd, pocket_rmsd = compute_protein_RMSD(pdbFile, p, ref_ligand_coords=ref_ligand_coords, gap_mask=gap_mask)
            with time_limit(60):
                result = compute_lddt(p, pdbFile, need_alignment=False, per_res=None, binding_site=ref_ligand_coords, gap_mask=gap_mask)
            lddt = result['lddt']
            bs_lddt = result['bs_lddt']
        except Exception as e:
            print("compute_protein_RMSD or lddt error", e, p, l)
            protein_rmsd = pocket_rmsd = -1
            lddt = bs_lddt = -1
        try:
            tmscore = compute_tmscore(p, pdbFile)['tmscore']
        except Exception as e:
            print(e)
            if str(e) == "Timed out!":
                tmscore = -1
            else:
                # skip_pdb_list.append(entryName)
                print(e, "compute tmscore")
                continue
        info.append([conformation, rank, protein_rmsd, pocket_rmsd, clashScore, lddt, bs_lddt, tmscore, overlap_mean, clash_n, n, rmsd_sym, rmsd])

    info = pd.DataFrame(info, columns=['conformation', 'rank', 'protein_rmsd', 'pocket_rmsd',
                                       'clashScore', 'lddt', 'bs_lddt', 'tmscore', 'overlap_mean', 'clash_n', 'n', 'rmsd', 'rmsd_unsym'])
    info.to_csv(infoFile)

def check_if_should_skip(pre, runName):
    if os.path.exists(f"{pre}/info_{runName}.csv"):
        print(runName, "already processed, skip")
        return True
    if not os.path.exists(f"{pre}/affinity_{runName}.csv"):
        print(runName, "move and pre-process first, skip")
        return True
    return False

def get_protein_rmsd_and_clash(data, with_gap_mask=False):
    info = []
    # for i, line in tqdm(data.iterrows(), total=len(data)):
    for i, line in data.iterrows():
        entryName = line['entryName']
        ligandFile = line['ligandFile']
        af2File = line['af2File']
        pdbFile = line['pdbFile']
        if with_gap_mask:
            gap_mask = line['gap_mask']
        else:
            gap_mask = None
        ref_mol = Chem.MolFromMolFile(ligandFile)
        ref_ligand_coords = ref_mol.GetConformer().GetPositions()

        clashScore, overlap, clash_n, n = compute_side_chain_metrics(af2File, ligandFile, verbose=True)
        overlap_mean = overlap.mean() if len(overlap) > 0 else 0

        protein_rmsd, pocket_rmsd = compute_protein_RMSD(pdbFile, af2File, ref_ligand_coords=ref_ligand_coords, gap_mask=gap_mask)
        result = compute_lddt(af2File, pdbFile, need_alignment=False, per_res=None, binding_site=ref_ligand_coords, gap_mask=gap_mask)
        lddt = result['lddt']
        bs_lddt = result['bs_lddt']
        try:
            tmscore = compute_tmscore(af2File, pdbFile)['tmscore']
        except Exception as e:
            print(e, i, line)
            if str(e) == "Timed out!":
                tmscore = -1
            else:
                # skip_pdb_list.append(entryName)
                continue
        info.append([entryName, protein_rmsd, pocket_rmsd, clashScore, lddt, bs_lddt, tmscore, overlap_mean, clash_n, n])

    info = pd.DataFrame(info, columns=['entryName', 'protein_rmsd', 'pocket_rmsd',
                                       'clashScore', 'lddt', 'bs_lddt', 'tmscore', 'overlap_mean', 'clash_n', 'n'])
    return info

# from pdbfixer import PDBFixer
# from openmm.app import PDBFile
def standardize_pdb(pdbFile, to_pdbFile):
    fixer = PDBFixer(filename=pdbFile)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    PDBFile.writeFile(fixer.topology, fixer.positions, open(to_pdbFile, 'w'), keepIds=True)


def get_database_dict():
    data_v11 = pd.read_csv('/mnt/nas/research-data/luwei/dynamicbind_data/pdbbind_v11/d3_with_clash_info.csv')
    v11_test = data_v11.query("group == 'test' and fraction_of_this_chain > 0.8").reset_index(drop=True)
    v11_test['name'] = v11_test['pdb']
    print("pdbbind_v11 test", data_v11.shape[0], v11_test.shape[0])


    data_v10 = pd.read_csv('/mnt/nas/research-data/luwei/dynamicbind_data/pdbbind_v10//d3_with_clash_info.csv')
    data_v10['entryName'] = data_v10['pdb']
    v10_test = data_v10.query("group == 'test'").reset_index(drop=True)
    v10_test['name'] = v10_test['pdb']
    print("pdbbind_v10 test", data_v10.shape[0], v10_test.shape[0])

    data_v9 = pd.read_csv('/gxr/luwei/dynamicbind/database/pdbbind_v9//data_with_ce_af2File_with_mutation.csv')
    data_v9['entryName'] = data_v9['pdb']
    v9_test = data_v9.query("group == 'test'").reset_index(drop=True)
    v9_test['name'] = v9_test['pdb']
    print("pdbbind_v9 test", data_v9.shape[0], v9_test.shape[0])


    pocketminer_v2 = pd.read_csv('/mnt/nas/research-data/luwei/dynamicbind_data/database/pocketminer_v2/d5_with_clash_info.csv')
    pocketminer_v2['uid'] = pocketminer_v2['uniprot_id']
    print("pocketminer_v2", len(pocketminer_v2))


    kinase = pd.read_csv("/mnt/nas/research-data/luwei/dynamicbind_data/case_studies/kinase_test_apr11.csv")
    kinase['entryName'] = kinase['name']
    print("kinase", len(kinase))

    highcite = pd.read_csv("/mnt/nas/research-data/luwei/dynamicbind_data/case_studies/highcite.csv")
    highcite['entryName'] = highcite['name']
    print("highcite", len(highcite))

    imatinib = pd.read_csv("/mnt/nas/research-data/luwei/dynamicbind_data/case_studies/manual_selected_imatinib.csv")
    imatinib['entryName'] = imatinib['name']
    print("imatinib", len(imatinib))

    pocketminer_apo = pd.read_csv('/mnt/nas/research-data/luwei/dynamicbind_data/database/pocketminer_apo//d4.csv')
    pocketminer_apo['name'] = pocketminer_apo['entryName']
    print("pocketminer_apo", len(pocketminer_apo))

    unseen = pd.read_csv("/mnt/nas/research-data/luwei/dynamicbind_data/case_studies/unseen.csv")
    unseen['name'] = unseen['entryName']
    print("unseen", len(unseen))

    new_binding_site = pd.read_csv("/mnt/nas/research-data/luwei/dynamicbind_data/new_binding_site.csv")
    new_binding_site['name'] = new_binding_site['entryName']
    print("new_binding_site", len(new_binding_site))

    kinase_has_clash = pd.read_csv("/mnt/nas/research-data/luwei/dynamicbind_data/whole_kinase_has_clash.csv")
    kinase_has_clash['name'] = kinase_has_clash['entryName']
    print("kinase_has_clash", len(kinase_has_clash))

    major_family = pd.read_csv("/mnt/nas/research-data/luwei/dynamicbind_data/major_family.csv")
    major_family['name'] = major_family['entryName']
    print("major_family", len(major_family))

    major_drug_targets = pd.read_csv( "/mnt/nas/research-data/luwei/dynamicbind_data/major_drug_targets.csv")
    major_drug_targets['name'] = major_drug_targets['entryName']
    print("major_drug_targets", len(major_drug_targets))

    database_dict = {'pdbbind_v9':v9_test, 'pocketminer_v2':pocketminer_v2,
                    'kinase':kinase, 'highcite':highcite, 'imatinib':imatinib, 'pdbbind_v10':v10_test,
                    'pdbbind_v11':v11_test, 'pocketminer_apo':pocketminer_apo,
                    'unseen':unseen, "new_binding_site":new_binding_site,
                     "kinase_has_clash":kinase_has_clash,
                     "major_family":major_family, "major_drug_targets":major_drug_targets,
                     }

    return database_dict

def get_summary(t):
    t = t.reset_index(drop=True)
    conformation_data_count = t.value_counts("conformation").to_dict()
    clash_mean = t.groupby(["conformation", "dataset_name"],sort=False)['clashScore'].mean().reset_index()
    below2A = t.groupby("conformation",sort=False).apply(lambda x: (x['rmsd']<2).sum() / len(x)).reset_index(name='below2A')
    below5A = t.groupby("conformation",sort=False).apply(lambda x: (x['rmsd']<5).sum() / len(x)).reset_index(name='below5A')
    pocket_rmsd = t.groupby("conformation",sort=False)['pocket_rmsd'].mean().reset_index()
    out = clash_mean.merge(below2A, on='conformation')
    out = out.merge(below5A, on='conformation')
    out = out.merge(pocket_rmsd, on='conformation')
    out['n_entries'] = out['conformation'].map(conformation_data_count)
    return out.round(3)

def combined_rank_selection(info):
    info['affinity_rank'] = info.groupby(["entryName", "conformation"],sort=False)['affinity'].rank(ascending=False)
    info['clash_rank'] = info.groupby(["entryName", "conformation"],sort=False)['clashScore'].rank()
    info['rmsd_combined_rank'] = info['rank']  + info['affinity_rank']/3 + info['clash_rank']
    select_by_affinity = info.loc[info.groupby(["entryName", "conformation"],
                                            sort=False)['rmsd_combined_rank'].agg('idxmin')].reset_index(drop=True)
    return select_by_affinity

def show_all_metrics(info, random_state=7):
    # selection method.
    select_by_clashScore = info.loc[info.groupby(["entryName", "conformation"],sort=False)['clashScore'].agg('idxmin')].reset_index(drop=True)
    a1 = get_summary(select_by_clashScore)
    a1['selection'] = 'by_clashScore'
    # a1['dataset_name'] = dataset_name
    # a1

    info['clash_rank'] = info.groupby(["entryName", "conformation"],sort=False)['clashScore'].rank()
    info['rmsd_r_p_r'] = info['rank'] / 5  + info['clash_rank']
    select_by_r_p_r = info.loc[info.groupby(["entryName", "conformation"],
                                            sort=False)['rmsd_r_p_r'].agg('idxmin')].reset_index(drop=True)
    a5 = get_summary(select_by_r_p_r)
    a5['selection'] = 'by_rank_div5_plus_clash_rank'
    # a5

    info['rank'] = info['rank'].astype(int)
    select_by_rank = info.loc[info.groupby(["entryName", "conformation"],
                                            sort=False)['rank'].agg('idxmin')].reset_index(drop=True)
    a4 = get_summary(select_by_rank)
    a4['selection'] = 'by_rank'
    # a4
    select_by_rmsd = info.loc[info.groupby(["entryName", "conformation"],sort=False)['rmsd'].agg('idxmin')].reset_index(drop=True)
    a2 = get_summary(select_by_rmsd)
    a2['selection'] = 'by_rmsd'
    # a2


    select_by_random = info.groupby(["entryName", "conformation"],sort=False).sample(1, random_state=random_state)
    a6 = get_summary(select_by_random)
    a6['selection'] = 'random'
    # a6

    select_by_affinity = combined_rank_selection(info)
    a7 = get_summary(select_by_affinity)
    a7['selection'] = 'rmsd_combined_rank'


    info['clash_rank'] = info.groupby(["entryName", "conformation"],sort=False)['clashScore'].rank()
    info['rmsd_r_p_r'] = info['rank']  + info['clash_rank'] / 2
    select_by_r_p_r = info.loc[info.groupby(["entryName", "conformation"],
                                            sort=False)['rmsd_r_p_r'].agg('idxmin')].reset_index(drop=True)
    a8 = get_summary(select_by_r_p_r)
    a8['selection'] = 'by_rank_plus_clash_rank_div2'
    a = pd.concat([a5, a8, a4, a2, a1, a6, a7]).reset_index(drop=True)
    return a
