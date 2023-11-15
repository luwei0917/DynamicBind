import copy

import numpy as np
from  scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader, DataListLoader

from tqdm import tqdm

from confidence.dataset import ListDataset
from utils import so3, torus
from utils.sampling import randomize_position, sampling
import torch
from utils.diffusion_utils import get_t_schedule, set_time

from torch_scatter import scatter_mean

def loss_function(lddt_pred, affinity_pred, tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred, res_chi_pred, data, t_to_sigma, device, lddt_weight=1, affinity_weight=1, tr_weight=1, rot_weight=1,
                  tor_weight=1, res_tr_weight=1, res_rot_weight=1, res_chi_weight=1, apply_mean=True, no_torsion=False, train_score=False, finetune=False):
    mean_dims = (0, 1) if apply_mean else 1
    if finetune:
        affinity = torch.cat([d.affinity for d in data], dim=0) if device.type == 'cuda' else data.affinity
        affinity_loss = ((affinity_pred.cpu() - affinity) ** 2).mean(dim=mean_dims) * 100.
        # affinity_loss = (((affinity_pred.cpu() - affinity) ** 2 * native_mask + torch.nn.ReLU()(affinity_pred.cpu() - affinity + 1.) ** 2 * (1.-native_mask))).mean(dim=mean_dims)
        affinity_base_loss = ((3.-affinity) ** 2).mean(dim=mean_dims).detach() * 100.
        return affinity_loss, affinity_loss.detach(), affinity_base_loss.detach()
    data_t = [torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
      for noise_type in ['tr', 'rot', 'tor', 'res_tr', 'res_rot', 'res_chi']]
    tr_sigma, rot_sigma, tor_sigma, res_tr_sigma, res_rot_sigma, res_chi_sigma  = t_to_sigma(*data_t)
    # res_tr_sigma = res_tr_sigma * torch.cat([d['receptor'].af2_trans_sigma for d in data]) if device.type == 'cuda' else data['receptor'].af2_trans_sigma
    # res_rot_sigma = res_rot_sigma * torch.cat([d['receptor'].af2_rotvecs_sigma for d in data]) if device.type == 'cuda' else data['receptor'].af2_rotvecs_sigma
    if tr_pred.abs().max() > 100:
        print([d.name for d in data])
        print(tr_pred)
        print(rot_pred)
    # lddt and affinity component
    lddt = torch.cat([d.lddt for d in data], dim=0) if device.type == 'cuda' else data.lddt

    lddt_loss = ((lddt_pred.cpu() - lddt) ** 2).mean(dim=mean_dims)
    lddt_base_loss = (lddt ** 2).mean(dim=mean_dims).detach()
    # native_mask = (lddt > 0.9).float()
    affinity = torch.cat([d.affinity for d in data], dim=0) if device.type == 'cuda' else data.affinity
    affinity_mask = (affinity != -1).float()
    affinity_loss = ((((affinity_pred.cpu() - affinity) ** 2)*affinity_mask) / (affinity_mask+1e-6)).mean(dim=mean_dims)
    # affinity_loss = (((affinity_pred.cpu() - affinity) ** 2 * native_mask + torch.nn.ReLU()(affinity_pred.cpu() - affinity + 1.) ** 2 * (1.-native_mask))).mean(dim=mean_dims)
    affinity_base_loss = (((affinity ** 2)*affinity_mask) / (affinity_mask+1e-6)).mean(dim=mean_dims)

    if finetune:
        if not train_score:
            loss = lddt_loss * lddt_weight + affinity_loss * affinity_weight
            base_loss = lddt_base_loss * lddt_weight + affinity_base_loss * affinity_weight

            return loss, lddt_loss.detach(), affinity_loss.detach(), 0., 0., 0., 0., 0., 0.,\
                    base_loss,lddt_base_loss, affinity_base_loss, 0., 0., 0., 0., 0., 0.

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score) ** 2 / tr_sigma ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 / tr_sigma ** 2).mean(dim=mean_dims).detach()


    # rotation component
    # rot_loss_weight = torch.cat([d.rot_loss_weight for d in data], dim=0) if device.type == 'cuda' else data.rot_loss_weight
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_pred_norm = rot_pred.norm(dim=-1,keepdim=True).cpu()
    rot_pred_vec = rot_pred.cpu() / (rot_pred_norm+1e-12)

    rot_loss_pos = (((rot_pred.cpu() - rot_score) / rot_sigma[...,None]) ** 2).mean(dim=1)
    rot_loss_neg = ((((rot_pred_norm-2*np.pi)*rot_pred_vec - rot_score) / rot_sigma[...,None]) ** 2).mean(dim=1)
    rot_loss = torch.minimum(rot_loss_pos,rot_loss_neg)

    if apply_mean:
        rot_loss = rot_loss.mean()
    rot_loss = rot_loss
    rot_base_loss = ((rot_score / rot_sigma[...,None]) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion and len(tor_pred) > 0:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge)).float()
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        # tor_loss_weight = torch.cat([d.tor_loss_weight for d in data], dim=0) if device.type == 'cuda' else data.tor_loss_weight
        tor_loss = ((1-(tor_pred.cpu() - tor_score).cos()) / (edge_tor_sigma/np.pi))
        tor_base_loss = ((1-(tor_score).cos()) / (edge_tor_sigma/np.pi))
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    res_decay_weight = torch.cat([d.res_decay_weight for d in data], dim=0) if device.type == 'cuda' else data.res_decay_weight
    res_gap_masks = torch.cat([d.gap_masks for d in data], dim=0) if device.type == 'cuda' else data.gap_masks
    # res_loss_weight = torch.cat([d.res_loss_weight for d in data], dim=0) if device.type == 'cuda' else data.res_loss_weight
    res_decay_weight = res_decay_weight * (1.-res_gap_masks)
    res_decay_weight = res_decay_weight / res_decay_weight.sum() * (1.-res_gap_masks).sum()
    res_loss_weight = torch.cat([d.res_loss_weight for d in data], dim=0) if device.type == 'cuda' else data.res_loss_weight
    res_loss_weight = res_decay_weight  * res_loss_weight


    # local translation component
    res_tr_score = torch.cat([d.res_tr_score for d in data], dim=0) if device.type == 'cuda' else data.res_tr_score
    res_tr_loss = torch.nn.L1Loss(reduction='none')(res_tr_pred.cpu(),res_tr_score).mean(dim=1) * res_loss_weight.squeeze(1) * 3.0#((res_tr_pred.cpu() - res_tr_score) ** 2).mean(dim=mean_dims)
    res_tr_base_loss = (res_tr_score).abs().mean(dim=1).detach() * res_loss_weight.squeeze(1) * 3.0
    if apply_mean:
        res_tr_loss = res_tr_loss.mean()
        res_tr_base_loss = res_tr_base_loss.mean()

    # local rotation component
    res_rot_score = torch.cat([d.res_rot_score for d in data], dim=0) if device.type == 'cuda' else data.res_rot_score

    # res_rot_pred_norm = res_rot_pred.norm(dim=-1,keepdim=True).cpu()
    # res_rot_pred_vec = res_rot_pred.cpu() / (res_rot_pred_norm+1e-12)

    res_rot_loss_pos = (torch.nn.L1Loss(reduction='none')(res_rot_pred.cpu(),res_rot_score)).mean(dim=1)
    # res_rot_loss_neg = (torch.nn.L1Loss(reduction='none')((res_rot_pred_norm-2*np.pi)*res_rot_pred_vec,res_rot_score)).mean(dim=1)
    # res_rot_loss = torch.minimum(res_rot_loss_pos,res_rot_loss_neg)
    res_rot_loss = res_rot_loss_pos * res_loss_weight.squeeze(1) * 15.0
    res_rot_base_loss = (res_rot_score.abs()).mean(dim=1).detach() * res_loss_weight.squeeze(1) * 15.0
    if apply_mean:
        res_rot_loss = res_rot_loss.mean()
        res_rot_base_loss = res_rot_base_loss.mean()

    res_chi_score = torch.cat([d.res_chi_score for d in data], dim=0) if device.type == 'cuda' else data.res_chi_score
    res_chi_mask = torch.cat([d['receptor'].chi_masks for d in data], dim=0) if device.type == 'cuda' else data['receptor'].chi_masks
    res_chi_mask = res_chi_mask[:,[0,2,4,5,6]]
    res_chi_symmetry_mask = torch.cat([d['receptor'].chi_symmetry_masks for d in data], dim=0) if device.type == 'cuda' else data['receptor'].chi_symmetry_masks
    res_chi_symmetry_mask = res_chi_symmetry_mask.bool()
    res_chi_loss = 1-(res_chi_pred.cpu()-res_chi_score).cos()
    res_chi_symmetry_loss = 1-(res_chi_pred.cpu()-res_chi_score-np.pi).cos()
    res_chi_loss[res_chi_symmetry_mask] = torch.minimum(res_chi_loss[res_chi_symmetry_mask],res_chi_symmetry_loss[res_chi_symmetry_mask])
    res_chi_loss = (res_chi_loss*res_loss_weight*res_chi_mask).sum(dim=mean_dims) / (res_chi_mask.sum(dim=mean_dims)+1e-12) * 3.0
    res_chi_base_loss = ((1-(res_chi_score).cos())*res_loss_weight*res_chi_mask).sum(dim=mean_dims) / (res_chi_mask.sum(dim=mean_dims).detach()+1e-12) * 3.0
    if not apply_mean:
        rec_batch = torch.cat([torch.tensor([i]*d['receptor'].num_nodes) for i,d in enumerate(data)], dim=0) if device.type == 'cuda' else data['receptor'].batch
        res_tr_loss = scatter_mean(res_tr_loss, rec_batch)
        res_rot_loss = scatter_mean(res_rot_loss, rec_batch)
        res_chi_loss = scatter_mean(res_chi_loss, rec_batch)
        res_tr_base_loss = scatter_mean(res_tr_base_loss, rec_batch)
        res_rot_base_loss = scatter_mean(res_rot_base_loss, rec_batch)
        res_chi_base_loss = scatter_mean(res_chi_base_loss, rec_batch)
    loss = lddt_loss * lddt_weight + affinity_loss * affinity_weight + tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + res_tr_loss * res_tr_weight + res_rot_loss * res_rot_weight + res_chi_loss * res_chi_weight
    base_loss = lddt_base_loss * lddt_weight + affinity_base_loss * affinity_weight + tr_base_loss * tr_weight + rot_base_loss * rot_weight + tor_base_loss * tor_weight + res_tr_base_loss * res_tr_weight + res_rot_base_loss * res_rot_weight + res_chi_base_loss * res_chi_weight

    return loss, lddt_loss.detach(), affinity_loss.detach(), tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), res_tr_loss.detach(), res_rot_loss.detach(), res_chi_loss.detach(),\
            base_loss,lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights, train_score=False, finetune=False):
    model.train()
    meter = AverageMeter(['loss', 'lddt_loss', 'affinity_loss', 'tr_loss', 'rot_loss', 'tor_loss', 'res_tr_loss', 'res_rot_loss', 'res_chi_loss', 'base_loss', 'lddt_base_loss', 'affinity_base_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'res_tr_base_loss', 'res_rot_base_loss', 'res_chi_base_loss'])

    bar = tqdm(loader, total=len(loader))
    train_loss = 0.0
    train_num = 0.0
    for data in bar:
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            lddt_pred, affinity_pred, tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred, res_chi_pred = model(data)
            loss, lddt_loss, affinity_loss, tr_loss, rot_loss, tor_loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss = \
                loss_fn(lddt_pred, affinity_pred, tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred, res_chi_pred, data=data, t_to_sigma=t_to_sigma, device=device, train_score=train_score, finetune=finetune)
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            ema_weights.update(model.parameters())
            meter.add([loss.cpu().detach(), lddt_loss, affinity_loss, tr_loss, rot_loss, tor_loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss])
            train_loss += loss.item()
            train_num += 1
            bar.set_description('loss: %.4f' % (train_loss/train_num))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del lddt_pred, affinity_pred, tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred, res_chi_pred
                    del loss, lddt_loss, affinity_loss, tr_loss, rot_loss, tor_loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del lddt_pred, affinity_pred, tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred, res_chi_pred
                    del loss, lddt_loss, affinity_loss, tr_loss, rot_loss, tor_loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'no cross edge found' in str(e):
                print('| WARNING: no cross edge found, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del lddt_pred, affinity_pred, tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred, res_chi_pred
                    del loss, lddt_loss, affinity_loss, tr_loss, rot_loss, tor_loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                # raise e
                continue
    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'lddt_loss', 'affinity_loss', 'tr_loss', 'rot_loss', 'tor_loss', 'res_tr_loss', 'res_rot_loss', 'res_chi_loss', 'base_loss', 'lddt_base_loss', 'affinity_base_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'res_tr_base_loss', 'res_rot_base_loss', 'res_chi_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'lddt_loss', 'affinity_loss', 'tr_loss', 'rot_loss', 'tor_loss', 'res_tr_loss', 'res_rot_loss', 'res_chi_losss', 'base_loss', 'lddt_base_loss', 'affinity_base_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'res_tr_base_loss', 'res_rot_base_loss', 'res_chi_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                lddt_pred, affinity_pred, tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred, res_chi_pred = model(data)

            loss, lddt_loss, affinity_loss, tr_loss, rot_loss, tor_loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss = \
                loss_fn(lddt_pred, affinity_pred, tr_pred, rot_pred, tor_pred, res_tr_pred, res_rot_pred, res_chi_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            # print(loss)
            meter.add([loss.cpu().detach(), lddt_loss, affinity_loss, tr_loss, rot_loss, tor_loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor, complex_t_res_tr, complex_t_res_rot, complex_t_res_chi = [torch.cat([d.complex_t[noise_type] for d in data]) for
                                                              noise_type in ['tr', 'rot', 'tor', 'res_tr', 'res_rot', 'res_chi']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                sigma_index_res_tr = torch.round(complex_t_res_tr.cpu() * (10 - 1)).long()
                sigma_index_res_rot = torch.round(complex_t_res_rot.cpu() * (10 - 1)).long()
                sigma_index_res_chi = torch.round(complex_t_res_chi.cpu() * (10 - 1)).long()
                meter_all.add(
                    [loss.cpu().detach(), lddt_loss, affinity_loss, tr_loss, rot_loss, tor_loss, res_tr_loss, res_rot_loss, res_chi_loss, base_loss, lddt_base_loss, affinity_base_loss, tr_base_loss, rot_base_loss, tor_base_loss, res_tr_base_loss, res_rot_base_loss, res_chi_base_loss],
                    [sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_tr, sigma_index_rot,
                     sigma_index_tor, sigma_index_tr, sigma_index_tr, sigma_index_tr])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'no cross edge found' in str(e):
                print('| WARNING: no cross edge found - skipping batch')
                continue
            else:
                raise e

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out

def inference_epoch(model, complex_graphs, device, t_to_sigma, args):
    model.eval()
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule, res_tr_schedule, res_rot_schedule, res_chi_schedule = t_schedule, t_schedule, t_schedule, t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    all_lddt = []
    all_lddt_pred = []
    all_affinity = []
    all_affinity_pred = []
    rmsds = []

    # n_batch = int(np.ceil(len(complex_graphs)/args.sample_batch_size))
    si = 0
    orig_complex_graphs = []
    data_list = []
    for orig_complex_graph in tqdm(loader):
        orig_complex_graphs.append(orig_complex_graph)
        data_list.append(copy.deepcopy(orig_complex_graph))
        si += 1
        if (si+1) % args.sample_batch_size != 0 and si != len(complex_graphs):
            continue
        elif len(data_list) > 0:
            randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max, args.rot_sigma_max, args.tor_sigma_max, args.res_tr_sigma_max, args.res_rot_sigma_max)

            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list == None:
                try:
                    predictions_list, lddt_pred, affinity_pred = sampling(data_list=data_list, model=model.module if device.type=='cuda' else model,
                                                             inference_steps=args.inference_steps,ode=True,
                                                             tr_schedule=tr_schedule, rot_schedule=rot_schedule, tor_schedule=tor_schedule,
                                                             res_tr_schedule=res_tr_schedule, res_rot_schedule=res_rot_schedule, res_chi_schedule=res_chi_schedule,
                                                             device=device, t_to_sigma=t_to_sigma, model_args=args)
                except Exception as e:
                    if 'failed to converge' in str(e):
                        failed_convergence_counter += 1
                        if failed_convergence_counter > 5:
                            print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                            break
                        print('| WARNING: SVD failed to converge - trying again with a new sample')
                    elif 'no cross edge found' in str(e):
                        failed_convergence_counter += 1
                        if failed_convergence_counter > 5:
                            print('| WARNING: no cross edge found - skipping the complex')
                            break
                        print('| WARNING: no cross edge found - trying again with a new sample')
                    else:
                        raise e
            if failed_convergence_counter > 5:
                orig_complex_graphs = []
                data_list = []
                continue

            for i,data in enumerate(predictions_list):
                orig_complex_graph = orig_complex_graphs[i]
                orig_ca_lig_cross_distances = (orig_complex_graph['ligand'].pos[None,...] - orig_complex_graph['receptor'].pos[:,None,...]).norm(dim=-1)

                ca_lig_cross_distances = (data['ligand'].pos[None,...] - data['receptor'].pos[:,None,...]).norm(dim=-1)
                ca_lig_cross_distances_diff = (orig_ca_lig_cross_distances - ca_lig_cross_distances).abs()

                cutoff_mask = (orig_ca_lig_cross_distances < 15.0).float()
                score = 0.25 * ((ca_lig_cross_distances_diff<0.5).float()
                                + (ca_lig_cross_distances_diff<1.0).float()
                                + (ca_lig_cross_distances_diff<2.0).float()
                                + (ca_lig_cross_distances_diff<4.0).float())
                ca_lig_cross_lddt =  (score * cutoff_mask).sum() / (cutoff_mask.sum()+1e-12)
                lddt = ca_lig_cross_lddt.unsqueeze(0)
                all_lddt.append(lddt)
                all_lddt_pred.append(lddt_pred[i])
                if data.affinity != -1:
                    all_affinity.append(data.affinity)
                    all_affinity_pred.append(affinity_pred[i])
                if args.no_torsion:
                    orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() +
                                                             orig_complex_graph.original_center.cpu().numpy())

                filterHs = torch.not_equal(data['ligand'].x[:, 0], 0).cpu().numpy()

                if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                    orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

                ligand_pos = []
                rec_pos = orig_complex_graph['receptor'].pos.cpu().numpy()
                pred_rec_pos = data['receptor'].pos.cpu().numpy()
                tran,rot = get_align_rotran(pred_rec_pos,rec_pos)
                ligand_pos.append(data['ligand'].pos.cpu().numpy()[filterHs]@rot+tran)
                ligand_pos = np.asarray(ligand_pos)
                # ligand_pos = np.asarray(
                #     [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
                orig_ligand_pos = np.expand_dims(
                    orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
                rmsds.append(rmsd)
            orig_complex_graphs = []
            data_list = []

    if len(all_lddt) == 0:
        losses = {'lddt_rmse': 0,
                  'lddt_base_rmse': 0,
                  'lddt_pearson': 0,
                  'lddt_spearman': 0,
                  'affinity_rmse': 0,
                  'affinity_base_rmse': 0,
                  'affinity_pearson': 0,
                  'affinity_spearman': 0,
                  'rmsds_lt2': 0,
                  'rmsds_lt5': 0}
        return losses

    all_lddt = torch.cat(all_lddt).view(-1).cpu().numpy()
    all_lddt_pred = torch.cat(all_lddt_pred).view(-1).cpu().numpy()

    # all_affinity_pred = np.minimum(torch.cat(all_affinity_pred).view(-1).cpu().numpy() / (all_lddt_pred+1e-12),15.)

    lddt_rmse = np.sqrt(((all_lddt-all_lddt_pred)**2).mean())
    lddt_base_rmse = np.sqrt(((all_lddt-all_lddt.mean())**2).mean())
    lddt_pearson = pearsonr(all_lddt, all_lddt_pred)[0]
    lddt_spearman = spearmanr(all_lddt, all_lddt_pred)[0]
    if len(all_affinity) > 0:
        all_affinity = torch.cat(all_affinity).view(-1).cpu().numpy()
        all_affinity_pred = torch.cat(all_affinity_pred).view(-1).cpu().numpy()
        affinity_rmse = np.sqrt(((all_affinity-all_affinity_pred)**2).mean())
        affinity_base_rmse = np.sqrt(((all_affinity-all_affinity.mean())**2).mean())
        affinity_pearson = pearsonr(all_affinity, all_affinity_pred)[0]
        affinity_spearman = spearmanr(all_affinity, all_affinity_pred)[0]
    else:
        affinity_rmse = 0.
        affinity_base_rmse = 0.
        affinity_pearson = 0.
        affinity_spearman = 0.
    rmsds = np.array(rmsds)
    losses = {'lddt_rmse': lddt_rmse,
              'lddt_base_rmse': lddt_base_rmse,
              'lddt_pearson': lddt_pearson,
              'lddt_spearman': lddt_spearman,
              'affinity_rmse': affinity_rmse,
              'affinity_base_rmse': affinity_base_rmse,
              'affinity_pearson': affinity_pearson,
              'affinity_spearman': affinity_spearman,
              'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))}
    return losses

def finetune_epoch(model, loader, device, t_to_sigma, args, optimizer, loss_fn, ema_weights):
    model.train()
    meter = AverageMeter(['loss', 'affinity_loss', 'affinity_base_loss'])

    bar = tqdm(loader, total=len(loader))
    train_loss = 0.0
    train_num = 0.0
    for data in bar:
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi = [0.6] * 6
            for d in data:
                set_time(d, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, 1, False, None)
            affinity_pred = model(data)
            # print(affinity_pred)
            loss, affinity_loss, affinity_base_loss = \
                loss_fn(None, affinity_pred, None, None, None, None, None, None, data=data, t_to_sigma=t_to_sigma, device=device, finetune=True)
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            ema_weights.update(model.parameters())
            meter.add([loss.cpu().detach(), affinity_loss, affinity_base_loss])
            train_loss += loss.item()
            train_num += 1
            bar.set_description('loss: %.4f' % (train_loss/train_num))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del affinity_pred
                    del loss, affinity_loss, affinity_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del affinity_pred
                    del loss, affinity_loss, affinity_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'no cross edge found' in str(e):
                print('| WARNING: no cross edge found, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del affinity_pred
                    del loss, affinity_loss, affinity_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                # raise e
                continue
    return meter.summary()


def finetune_test_epoch(model, loader, device, t_to_sigma, args, loss_fn):
    model.eval()
    meter = AverageMeter(['loss', 'affinity_loss', 'affinity_base_loss'])

    bar = tqdm(loader, total=len(loader))
    train_loss = 0.0
    train_num = 0.0
    for data in bar:
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        try:
            t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi = [0.6] * 6
            for d in data:
                set_time(d, t_tr, t_rot, t_tor, t_res_tr, t_res_rot, t_res_chi, 1, False, None)
            affinity_pred= model(data)
            loss, affinity_loss, affinity_base_loss = \
                loss_fn(None, affinity_pred, None, None, None, None, None, None, data=data, t_to_sigma=t_to_sigma, device=device, finetune=True)
            # with torch.autograd.detect_anomaly():
            meter.add([loss.cpu().detach(), affinity_loss, affinity_base_loss])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del affinity_pred
                    del loss, affinity_loss, affinity_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del affinity_pred
                    del loss, affinity_loss, affinity_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            elif 'no cross edge found' in str(e):
                print('| WARNING: no cross edge found, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                try:
                    del affinity_pred
                    del loss, affinity_loss, affinity_base_loss
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            else:
                print(e)
                # raise e
                continue
    return meter.summary()

from numpy import dot, transpose, sqrt
from numpy.linalg import svd, det

def get_align_rotran(coords,reference_coords):
    # center on centroid
    av1 = coords.mean(0,keepdims=True)
    av2 = reference_coords.mean(0,keepdims=True)
    coords = coords - av1
    reference_coords = reference_coords - av2
    # correlation matrix
    a = dot(transpose(coords), reference_coords)
    u, d, vt = svd(a)
    rot = transpose(dot(transpose(vt), transpose(u)))
    # check if we have found a reflection
    if det(rot) < 0:
        vt[2] = -vt[2]
        rot = transpose(dot(transpose(vt), transpose(u)))
    tran = av2 - dot(av1, rot)
    return tran, rot
