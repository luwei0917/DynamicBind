import copy
import os
import torch
import shutil
import warnings
warnings.filterwarnings("ignore")

import time
from argparse import ArgumentParser, Namespace, FileType
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
import scipy
from Bio.PDB import PDBParser

from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit import Chem

import torch
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')



from torch_geometric.loader import DataLoader


from datasets.process_mols import read_molecule, generate_conformer, write_mol_with_coords
from datasets.pdbbind import PDBBind,PDBBindScoring
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, set_time
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import LigandToPDB, modify_pdb, receptor_to_pdb, save_protein
from utils.clash import compute_side_chain_metrics
# from utils.relax import openmm_relax
from tqdm import tqdm
import datetime
from contextlib import contextmanager

from multiprocessing import Pool as ThreadPool

import random
import pickle
# pool = ThreadPool(8)

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
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path and --ligand parameters')
parser.add_argument('--protein_path', type=str, default='data/dummy_data/1a0q_protein.pdb', help='Path to the protein .pdb file')
parser.add_argument('--ligand', type=str, default='COc(cc1)ccc1C#N', help='Either a SMILES string or the path to a molecule file that rdkit can read')
parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--esm_embeddings_path', type=str, default='data/esm2_output', help='If this is set then the LM embeddings at that path will be used for the receptor features')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')
parser.add_argument('--savings_per_complex', type=int, default=1, help='Number of samples to save')
parser.add_argument('--seed', type=int, default=42, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--cache_path', type=str, default='data/cache', help='Folder from where to load/restore cached dataset')
parser.add_argument('--no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--ode', action='store_true', default=False, help='Use ODE formulation for inference')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='')
parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')
parser.add_argument('--keep_local_structures', action='store_true', default=False, help='Keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit')
parser.add_argument('--protein_dynamic', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--relax', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--use_existing_cache', action='store_true', default=False, help='Use existing cache file, if they exist.')


args = parser.parse_args()
def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(seed=args.seed)
if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value

os.makedirs(args.out_dir, exist_ok=True)

with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))

if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    # df = df[:10]
    if 'crystal_protein_path' not in df.columns:
        df['crystal_protein_path'] = df['protein_path']
    protein_path_list = df['protein_path'].tolist()
    ligand_descriptions = df['ligand'].tolist()
    # if 'name' not in df.columns:
    #     df['name'] = [f'idx_{i}' for i in range(df.shape[0])]
    # elif df['name'].nunique() < df.shape[0]:
    #     df['name'] = [f'idx_{i}' for i in range(df.shape[0])]
    df['name'] = [f'idx_{i}' for i in range(df.shape[0])]
    name_list = df['name'].tolist()
else:
    protein_path_list = [args.protein_path]
    ligand_descriptions = [args.ligand]

test_dataset = PDBBindScoring(transform=None, root='', name_list=name_list, protein_path_list=protein_path_list, ligand_descriptions=ligand_descriptions,
                       receptor_radius=score_model_args.receptor_radius, cache_path=args.cache_path,
                       remove_hs=score_model_args.remove_hs, max_lig_size=None,
                       c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors, matching=False, keep_original=False,
                       popsize=score_model_args.matching_popsize, maxiter=score_model_args.matching_maxiter,center_ligand=True,
                       all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                       atom_max_neighbors=score_model_args.atom_max_neighbors,
                       esm_embeddings_path= args.esm_embeddings_path if score_model_args.esm_embeddings_path is not None else None,
                       require_ligand=True,require_receptor=True, num_workers=args.num_workers, keep_local_structures=args.keep_local_structures, use_existing_cache=args.use_existing_cache)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

if args.confidence_model_dir is not None:
    if confidence_args.transfer_weights:
        with open(f'{confidence_args.original_model_dir}/model_parameters.yml') as f:
            confidence_model_args = Namespace(**yaml.full_load(f))
    else:
        confidence_model_args = confidence_args
    confidence_model = get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
    state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
    confidence_model.load_state_dict(state_dict, strict=True)
    confidence_model = confidence_model.to(device)
    confidence_model.eval()
else:
    confidence_model = None
    confidence_args = None
    confidence_model_args = None

tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
rot_schedule = tr_schedule
tor_schedule = tr_schedule
res_tr_schedule = tr_schedule
res_rot_schedule = tr_schedule
res_chi_schedule = tr_schedule
print('common t schedule', tr_schedule)

failures, skipped, confidences_list, names_list, run_times, min_self_distances_list = 0, 0, [], [], [], []
N = args.samples_per_complex
print('Size of test dataset: ', len(test_dataset))

affinity_pred = {}
all_complete_affinity = []
for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
    # if idx not in [54, 123, 141, 157, 165, 251]:continue
    try:
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        randomize_position(data_list, score_model_args.no_torsion, args.no_random,score_model_args.tr_sigma_max,score_model_args.rot_sigma_max, score_model_args.tor_sigma_max,score_model_args.res_tr_sigma_max,score_model_args.res_rot_sigma_max)
        pdb = None
        lig = orig_complex_graph.mol[0]
        receptor_pdb = orig_complex_graph.rec_pdb[0]
        pdb_or_cif = receptor_pdb.get_full_id()[0]
        if score_model_args.remove_hs: lig = RemoveHs(lig)
        visualization_list = None

        start_time = time.time()
        confidence = None
        steps = args.actual_steps if args.actual_steps is not None else args.inference_steps
        final_data_list, data_list_step, all_lddt_pred, all_affinity_pred = [],[[] for _ in range(steps)],[],[]
        for i in range(int(np.ceil(len(data_list)/args.batch_size))):
            try:
                outputs = sampling(data_list=data_list[i*args.batch_size:(i+1)*args.batch_size], model=model,
                                 inference_steps=steps,
                                 tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule, res_tr_schedule=res_tr_schedule, res_rot_schedule=res_rot_schedule, res_chi_schedule=res_chi_schedule,
                                 device=device, t_to_sigma=t_to_sigma, model_args=score_model_args, no_random=args.no_random,
                                 ode=args.ode, visualization_list=visualization_list, batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise, protein_dynamic=args.protein_dynamic)
                all_lddt_pred.append(outputs[2])
                all_affinity_pred.append(outputs[3])
            except Exception as e:
                # raise e
                print(e)
        all_lddt_pred = torch.cat(all_lddt_pred)
        all_affinity_pred = torch.cat(all_affinity_pred)
        ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in final_data_list])
        final_receptor_pdbs = []

        # with Timer('modify pdb'):
        #     final_receptor_pdbs = pool.map(modify_pdb, zip([copy.deepcopy(receptor_pdb) for _ in range(len(data_list))], data_list))
        # run_times.append(time.time() - start_time)

        # sample_ligand_path_list = []
        # sample_protein_path_list = []
        # for rank, pos in enumerate(ligand_pos):
        #     mol_pred = copy.deepcopy(lig)
        #     if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
        #     write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_ligand.sdf'))
        #     save_protein(final_receptor_pdbs[rank],os.path.join(write_dir, f'rank{rank+1}_receptor.pdb'))
        #     sample_ligand_path_list.append(os.path.join(write_dir, f'rank{rank+1}_ligand.sdf'))
        #     sample_protein_path_list.append(os.path.join(write_dir, f'rank{rank+1}_receptor.pdb'))


        all_lddt_pred = all_lddt_pred.view(-1).cpu().numpy()
        # print(all_lddt_pred)
        all_affinity_pred = all_affinity_pred.view(-1).cpu().numpy()
        final_affinity_pred = np.minimum((all_affinity_pred*all_lddt_pred).sum() / (all_lddt_pred.sum()+1e-12),15.)

        affinity_pred[orig_complex_graph.name[0]] = final_affinity_pred

        complete_affinity = pd.DataFrame({'name':orig_complex_graph.name[0],'lddt':all_lddt_pred,'affinity':all_affinity_pred})
        all_complete_affinity.append(complete_affinity)
        names_list.append(orig_complex_graph.name[0])
    except Exception as e:
        # raise(e)
        print("Failed on", orig_complex_graph["name"], e)
        failures += 1

print(f'Failed for {failures} complexes')
print(f'Skipped {skipped} complexes')

affinity_pred_df = pd.DataFrame({'name':list(affinity_pred.keys()),'affinity':list(affinity_pred.values())})
affinity_pred_df.to_csv(f'{args.out_dir}/affinity_prediction.csv',index=False)
pd.concat(all_complete_affinity).to_csv(f'{args.out_dir}/complete_affinity_prediction.csv',index=False)

# min_self_distances = np.array(min_self_distances_list)
# confidences = np.array(confidences_list)
# names = np.array(names_list)
# run_times = np.array(run_times)
# np.save(f'{args.out_dir}/min_self_distances.npy', min_self_distances)
# np.save(f'{args.out_dir}/confidences.npy', confidences)
# np.save(f'{args.out_dir}/run_times.npy', run_times)
# np.save(f'{args.out_dir}/complex_names.npy', np.array(names))

print(f'Results are in {args.out_dir}')
