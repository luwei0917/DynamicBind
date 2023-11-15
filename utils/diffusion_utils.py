import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles

from utils.affine import T

def t_to_sigma(t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, args):
    tr_sigma = args.tr_sigma_min ** (1-t_tr) * args.tr_sigma_max ** t_tr
    rot_sigma = args.rot_sigma_min ** (1-t_rot) * args.rot_sigma_max ** t_rot
    tor_sigma = args.tor_sigma_min ** (1-t_tor) * args.tor_sigma_max ** t_tor

    if torch.is_tensor(t_tr):
        res_tr_sigma = torch.clamp(args.res_tr_sigma_min + (args.res_tr_sigma_max-args.res_tr_sigma_min) * (t_res_tr*5.) ** 0.3, max=torch.tensor(1.).float().to(t_res_tr.device)) #** (1-t_res_tr) * args.res_tr_sigma_max ** t_res_tr
        res_rot_sigma = torch.clamp(args.res_rot_sigma_min + (args.res_rot_sigma_max-args.res_rot_sigma_min) * (t_res_rot*5.) ** 0.3, max=torch.tensor(1.).float().to(t_res_rot.device)) #** (1-t_res_rot) * args.res_rot_sigma_max ** t_res_rot
        res_chi_sigma = torch.clamp(args.res_chi_sigma_min + (args.res_chi_sigma_max-args.res_chi_sigma_min) * (t_res_chi*5.) ** 0.3, max=torch.tensor(1.).float().to(t_res_chi.device))
    else:
        res_tr_sigma = min(args.res_tr_sigma_min + (args.res_tr_sigma_max-args.res_tr_sigma_min) * (t_res_tr*5.) ** 0.3, 1.) #** (1-t_res_tr) * args.res_tr_sigma_max ** t_res_tr
        res_rot_sigma = min(args.res_rot_sigma_min + (args.res_rot_sigma_max-args.res_rot_sigma_min) * (t_res_rot*5.) ** 0.3, 1.) #** (1-t_res_rot) * args.res_rot_sigma_max ** t_res_rot
        res_chi_sigma = min(args.res_chi_sigma_min + (args.res_chi_sigma_max-args.res_chi_sigma_min) * (t_res_chi*5.) ** 0.3, 1.)

    # res_tr_sigma = args.res_tr_sigma_min ** (1-t_res_tr) * args.res_tr_sigma_max ** (t_res_tr)
    # res_rot_sigma = args.res_rot_sigma_min ** (1-t_res_rot) * args.res_rot_sigma_max ** (t_res_rot)
    # res_chi_sigma = args.res_chi_sigma_min ** (1-t_res_chi) * args.res_chi_sigma_max ** (t_res_chi) #** (1-t_res_rot) * args.res_rot_sigma_max ** t_res_rot
    return tr_sigma, rot_sigma, tor_sigma, res_tr_sigma, res_rot_sigma, res_chi_sigma


def modify_conformer(data, tr_update, rot_update, torsion_updates, res_tr_update, res_rot_update, res_chi_update):

    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T + tr_update + lig_center
    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                           data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) else data['ligand'].mask_rotate[0],
                                                           torsion_updates).to(rigid_new_pos.device)
        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        data['ligand'].pos = aligned_flexible_pos
        if 'torsion_updates' in data:
            data.torsion_updates = data.torsion_updates + torch.from_numpy(torsion_updates).float()

    else:
        data['ligand'].pos = rigid_new_pos

    # print(data['receptor'].pos[:3])
    # print(data['receptor'].chis[:3])
    res_rot_mat = axis_angle_to_matrix(res_rot_update)
    # print(data['receptor'].lf_3pts.shape,res_rot_mat.shape,res_tr_update.shape)
    data['receptor'].lf_3pts = (data['receptor'].lf_3pts - data['receptor'].lf_3pts[:,[1],:]) @ res_rot_mat.transpose(1,2) + data['receptor'].lf_3pts[:,[1],:] + res_tr_update[:,None,:]
    data['receptor'].pos = data['receptor'].pos + res_tr_update
    if 'acc_pred_chis' in data['receptor'].keys():
        data['receptor'].acc_pred_chis = ((data['receptor'].acc_pred_chis + res_chi_update) * data['receptor'].chi_masks[:,[0,2,4,5,6]]) % (2*np.pi)

    res_chi_update = res_chi_update[:,[0,0,1,1,2,3,4]]
    data['receptor'].chis = ((data['receptor'].chis + res_chi_update) * data['receptor'].chi_masks) % (2*np.pi)
    min_chi1 = torch.minimum(data['receptor'].chis[:,0],data['receptor'].chis[:,1])
    max_chi1 = torch.maximum(data['receptor'].chis[:,0],data['receptor'].chis[:,1])
    data['receptor'].chis[:,0] = max_chi1
    data['receptor'].chis[:,1] = min_chi1
    min_chi2 = torch.minimum(data['receptor'].chis[:,2],data['receptor'].chis[:,3])
    max_chi2 = torch.maximum(data['receptor'].chis[:,2],data['receptor'].chis[:,3])
    data['receptor'].chis[:,2] = max_chi2
    data['receptor'].chis[:,3] = min_chi2
    # print(data['receptor'].pos[:3])
    # print(data['receptor'].chis[:3])
    return data


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]


def set_time(complex_graphs, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, batchsize, all_atoms, device):
    complex_graphs['ligand'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device)}
    complex_graphs['receptor'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device)}
    complex_graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),
                               'rot': t_rot * torch.ones(batchsize).to(device),
                               'tor': t_tor * torch.ones(batchsize).to(device),
                               'res_tr': t_res_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
                               'res_rot': t_res_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
                               'res_chi': t_res_chi * torch.ones(complex_graphs['receptor'].num_nodes).to(device)}
    if all_atoms:
        complex_graphs['atom'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device)}
