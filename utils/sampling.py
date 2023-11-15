import numpy as np
import copy
import torch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R

from utils.affine import T
from utils.geometry import axis_angle_to_matrix
from utils.visualise import modify_pdb

def randomize_position(data_list, no_torsion, no_random, tr_sigma_max, rot_sigma_max, tor_sigma_max, res_tr_sigma_max, res_rot_sigma_max):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph.torsion_updates = torch.from_numpy(torsion_updates).float()
            # torsion_updates = np.random.normal(loc=0.0, scale=rot_sigma_max, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                    complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0], torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        # rot_update = torch.normal(mean=0, std=rot_sigma_max, size=(1, 3))
        # random_rotation = axis_angle_to_matrix(rot_update.squeeze())
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()

        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            idx = np.random.randint(len(complex_graph['receptor'].pos))
            tr_update = complex_graph['receptor'].pos[idx]# + torch.normal(mean=0, std=15.0, size=(1, 3))
            # new_ligand_pos = complex_graph['ligand'].pos + tr_update
            # dist = (((new_ligand_pos[:,None,:] - complex_graph['receptor'].pos[None,...])**2).sum(dim=-1))**0.5
            # min_dist = dist.min()
            # while min_dist < 15 or min_dist > 25:
            #     idx = np.random.randint(len(complex_graph['receptor'].pos))
            #     tr_update = complex_graph['receptor'].pos[idx] + torch.normal(mean=0, std=15.0, size=(1, 3))
            #     new_ligand_pos = complex_graph['ligand'].pos + tr_update
            #     dist = (((new_ligand_pos[:,None,:] - complex_graph['receptor'].pos[None,...])**2).sum(dim=-1))**0.5
            #     min_dist = dist.min()
            # print(min_dist)
            # tr_update = torch.normal(mean=0, std=20.0, size=(1, 3))
            complex_graph['ligand'].pos += tr_update

        if 'af2_trans' in complex_graph['receptor'].keys():
            res_tr_update = complex_graph['receptor'].af2_trans
            res_rot_update = complex_graph['receptor'].af2_rotvecs
            res_chi_update = complex_graph['receptor'].af2_chis
            res_rot_mat = axis_angle_to_matrix(res_rot_update)
            # complex_graph.res_tr_update = res_tr_update
            # complex_graph.res_rot_update = res_rot_mat
            # complex_graph.res_chi_update = res_chi_update
            complex_graph['receptor'].lf_3pts = (complex_graph['receptor'].lf_3pts - complex_graph['receptor'].lf_3pts[:,[1],:]) @ res_rot_mat.transpose(1,2) + complex_graph['receptor'].lf_3pts[:,[1],:] + res_tr_update[:,None,:]
            complex_graph['receptor'].pos = complex_graph['receptor'].pos + res_tr_update
            complex_graph['receptor'].chis = ((complex_graph['receptor'].chis + res_chi_update) * complex_graph['receptor'].chi_masks) % (2*np.pi)
        else:
            complex_graph['receptor'].chis = (complex_graph['receptor'].chis * complex_graph['receptor'].chi_masks) % (2*np.pi)
        min_chi1 = torch.minimum(complex_graph['receptor'].chis[:,0],complex_graph['receptor'].chis[:,1])
        max_chi1 = torch.maximum(complex_graph['receptor'].chis[:,0],complex_graph['receptor'].chis[:,1])
        complex_graph['receptor'].chis[:,0] = max_chi1
        complex_graph['receptor'].chis[:,1] = min_chi1
        min_chi2 = torch.minimum(complex_graph['receptor'].chis[:,2],complex_graph['receptor'].chis[:,3])
        max_chi2 = torch.maximum(complex_graph['receptor'].chis[:,2],complex_graph['receptor'].chis[:,3])
        complex_graph['receptor'].chis[:,2] = max_chi2
        complex_graph['receptor'].chis[:,3] = min_chi2
        # res_tr_update = torch.normal(mean=0, std=res_tr_sigma_max, size=(complex_graph['receptor'].pos.shape[0], 3))
        # res_random_rotation = torch.normal(mean=0, std=res_rot_sigma_max, size=(complex_graph['receptor'].pos.shape[0], 3))#res_random_rotation / np.linalg.norm(res_random_rotation,axis=-1,keepdims=True) * np.random.normal(0, res_rot_sigma_max/10, size=(complex_graph['receptor'].pos.shape[0], 1))
        # res_rot_mat = axis_angle_to_matrix(res_random_rotation)
        # # #
        # complex_graph['receptor'].lf_3pts = (complex_graph['receptor'].lf_3pts - complex_graph['receptor'].lf_3pts[:,[1],:]) @ res_rot_mat.transpose(1,2) + complex_graph['receptor'].lf_3pts[:,[1],:] + res_tr_update[:,None,:]
        # complex_graph['receptor'].pos = complex_graph['receptor'].pos + res_tr_update

def pred_lddt_and_affinity(complex_graph_batch, model, batch_size, device, model_args):
    all_lddt_pred = []
    all_affinity_pred = []
    # loader = DataLoader(data_list, batch_size=batch_size)
    # for complex_graph_batch in loader:
    b = complex_graph_batch.num_graphs
    t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi = [0.6] * 6
    complex_graph_batch = complex_graph_batch.to(device)
    set_time(complex_graph_batch, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, b, model_args.all_atoms, device)
    with torch.no_grad():
        lddt_pred, affinity_pred, tr_score, rot_score, tor_score, res_tr_score, res_rot_score, res_chi_score = model(complex_graph_batch)
    all_lddt_pred.append(lddt_pred)
    all_affinity_pred.append(affinity_pred)
    all_lddt_pred = torch.cat(all_lddt_pred,dim=0)
    all_affinity_pred = torch.cat(all_affinity_pred,dim=0)
    return all_lddt_pred, all_affinity_pred

def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, res_tr_schedule, res_rot_schedule, res_chi_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=True, visualization_list=None, confidence_model=None, batch_size=32, no_final_step_noise=False, return_per_step=False, protein_dynamic=True):
    N = len(data_list)
    data_list_step = []
    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx], res_tr_schedule[t_idx], res_rot_schedule[t_idx], res_chi_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]
        dt_res_tr = res_tr_schedule[t_idx] - res_tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else res_tr_schedule[t_idx]
        dt_res_rot = res_rot_schedule[t_idx] - res_rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else res_rot_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            n = complex_graph_batch['receptor'].pos.shape[0]
            complex_graph_batch = complex_graph_batch.to(device)
            tr_sigma, rot_sigma, tor_sigma, res_tr_sigma, res_rot_sigma, res_chi_sigma = t_to_sigma(t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, b, model_args.all_atoms, device)

            with torch.no_grad():
                lddt_pred, affinity_pred, tr_score, rot_score, tor_score, res_tr_score, res_rot_score, res_chi_score = model(complex_graph_batch)
            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            tr_f = (tr_g/tr_sigma) ** 2 * dt_tr
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))
            rot_f = dt_rot * (rot_g/rot_sigma) ** 2
            if ode:
                # tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()#
                # rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()#

                tr_perturb = torch.clamp(tr_score.cpu(), min=-20, max=20)#(inference_steps-t_idx) #* model_args.tr_sigma_max / inference_steps #+ torch.normal(mean=0, std=tr_sigma, size=(b, 3))  / (1+t_idx) #(inference_steps-t_idx)#
                rot_perturb = rot_score.cpu()#+ torch.normal(mean=0, std=1, size=(b, 3)) / (1+t_idx)#
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                # tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()
                tr_perturb = torch.clamp(tr_score.cpu()+tr_g*np.sqrt(dt_tr)*tr_z, min=-20, max=20)
                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                # rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()
                rot_perturb = rot_score.cpu() + rot_g * np.sqrt(dt_rot) * rot_z
            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                tor_f = (tor_g/tor_sigma) ** 2 * dt_tor
                if ode:
                    # tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                    tor_perturb = tor_score.cpu().numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    # tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                    tor_perturb = (tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                torsions_per_molecule = tor_perturb.shape[0] // b
            else:
                tor_perturb = None

            res_tr_g = 3*torch.sqrt(torch.tensor(2 * np.log(model_args.res_tr_sigma_max / model_args.res_tr_sigma_min)))
            res_rot_g = 3*torch.sqrt(torch.tensor(2 * np.log(model_args.res_rot_sigma_max / model_args.res_rot_sigma_min)))
            if ode or 1:
                if tr_sigma < 6 and protein_dynamic:
                    res_tr_perturb = res_tr_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                    res_rot_perturb = res_rot_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                    res_chi_perturb = res_chi_score.cpu() / (inference_steps-t_idx+inference_steps*0.25)
                    # res_tr_perturb = res_tr_score.cpu() / (t_idx+inference_steps*0.1)
                    # res_rot_perturb = res_rot_score.cpu() / (t_idx+inference_steps*0.1)
                    # res_chi_perturb = res_chi_score.cpu() / (t_idx+inference_steps*0.1)
                else:
                    res_tr_perturb = torch.zeros((n, 3))
                    res_rot_perturb = torch.zeros((n, 3))
                    res_chi_perturb = torch.zeros((n, 5))
                # if t_idx <= inference_steps - 1:
                #     res_tr_perturb = res_tr_score.cpu() / inference_steps * 10#(0.5 * res_tr_g ** 2 * dt_res_tr * res_tr_score.cpu()).cpu()
                #     res_rot_perturb = res_rot_score.cpu() / inference_steps * 10#(0.5 * res_rot_score.cpu() * dt_res_rot * res_rot_g ** 2).cpu()
                # else:
                # res_tr_perturb = torch.zeros((n, 3))
                # res_rot_perturb = torch.zeros((n, 3))
                # res_chi_perturb = torch.zeros((n, 5))
            else:
                # res_tr_z = torch.zeros((n, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                #     else torch.normal(mean=0, std=model_args.res_tr_sigma_max/20, size=(n, 3))
                # res_tr_perturb = (res_tr_g ** 2 * dt_res_tr * res_tr_score.cpu() + res_tr_g * np.sqrt(dt_res_tr) * res_tr_z).cpu()
                #
                # res_rot_z = torch.zeros((n, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                #     else torch.normal(mean=0, std=model_args.res_rot_sigma_max/20, size=(n, 3))
                # res_rot_perturb = (res_rot_score.cpu() * dt_res_rot * res_rot_g ** 2 + res_rot_g * np.sqrt(dt_res_rot) * res_rot_z).cpu()
                res_tr_perturb = torch.zeros((n, 3))
                res_rot_perturb = torch.zeros((n, 3))
                res_chi_z = torch.zeros((n, 5)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(n, 5))
                res_chi_perturb = res_chi_score.cpu() + res_chi_z*dt_res_chi

            res_tr_perturb = torch.clamp(res_tr_perturb, min=-20, max=20)       # safe perturb
            res_per_molecule = res_tr_perturb.shape[0] // b
            # Apply denoise
            # print(tr_perturb.shape,rot_perturb.shape,res_tr_perturb.shape,res_rot_perturb.shape)
            tor_i = 0
            res_i = 0
            for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list()):
                new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                              tor_perturb[tor_i:tor_i+complex_graph['ligand'].edge_mask.sum()] if not model_args.no_torsion else None,
                                              res_tr_perturb[res_i:res_i+complex_graph['receptor'].pos.shape[0]], res_rot_perturb[res_i:res_i+complex_graph['receptor'].pos.shape[0]],
                                              res_chi_perturb[res_i:res_i+complex_graph['receptor'].pos.shape[0]])])
                tor_i += complex_graph['ligand'].edge_mask.sum()
                res_i += complex_graph['receptor'].pos.shape[0]

        data_list = new_data_list
        data_list_step.append(new_data_list)
        # if visualization_list is not None:
        #     for idx, visualization in enumerate(visualization_list):
        #         visualization[0].add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
        #                           part=1, order=t_idx + 2)
        #         new_receptor_pdb = copy.deepcopy(visualization[1])
        #         if protein_dynamic:
        #             modify_pdb(new_receptor_pdb,data_list[idx])
        #         new_receptor_pdb.id = t_idx + 2
        #         visualization[2].add(new_receptor_pdb)
    all_lddt_pred = []
    all_affinity_pred = []
    loader = DataLoader(data_list, batch_size=batch_size)
    for complex_graph_batch in loader:
        b = complex_graph_batch.num_graphs
        t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi = [0.6] * 6
        complex_graph_batch = complex_graph_batch.to(device)
        set_time(complex_graph_batch, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, b, model_args.all_atoms, device)
        with torch.no_grad():
            lddt_pred, affinity_pred, tr_score, rot_score, tor_score, res_tr_score, res_rot_score, res_chi_score = model(complex_graph_batch)
        all_lddt_pred.append(lddt_pred)
        all_affinity_pred.append(affinity_pred)
    all_lddt_pred = torch.cat(all_lddt_pred,dim=0)
    all_affinity_pred = torch.cat(all_affinity_pred,dim=0)
    # all_lddt_pred, all_affinity_pred = pred_lddt_and_affinity(data_list, model, batch_size, device, model_args)
    return data_list, data_list_step, all_lddt_pred, all_affinity_pred
