import copy
import math
import os
from functools import partial
import numpy as np
# import wandb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torch_geometric.nn import DataParallel

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from datasets.pdbbind import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, finetune_epoch, inference_epoch
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage

gpus = list(range(torch.cuda.device_count()))
print('Available GPU count:',len(gpus))
def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, start_epoch):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    loss_fn = partial(loss_function, lddt_weight=args.lddt_weight, affinity_weight=args.affinity_weight, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                      tor_weight=args.tor_weight, res_tr_weight=args.res_tr_weight, res_rot_weight=args.res_rot_weight, res_chi_weight=args.res_chi_weight,
                      no_torsion=args.no_torsion)

    print("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch < start_epoch:
            continue
        if epoch % 5 == 0: print("Run name: ", args.run_name)
        logs = {}

        if not args.only_test:

            train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn, ema_weights)
            print("Epoch {}: Training loss {:.4f}  lddt {:.4f}  affinity {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  res_tr {:.4f}   res_rot {:.4f}   res_chi {:.4f}"
                  .format(epoch, train_losses['loss'], train_losses['lddt_loss'], train_losses['affinity_loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                          train_losses['tor_loss'], train_losses['res_tr_loss'], train_losses['res_rot_loss'], train_losses['res_chi_loss']))

            print("Epoch {}: Training base loss {:.4f} lddt {:.4f}  affinity {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  res_tr {:.4f}   res_rot {:.4f}   res_chi {:.4f}"
                      .format(epoch, train_losses['base_loss'], train_losses['lddt_base_loss'], train_losses['affinity_base_loss'], train_losses['tr_base_loss'], train_losses['rot_base_loss'],
                              train_losses['tor_base_loss'], train_losses['res_tr_base_loss'], train_losses['res_rot_base_loss'], train_losses['res_chi_base_loss']))

            if args.finetune_freq != None and (epoch + 1) % args.finetune_freq == 0 and best_val_inference_value > 20.:
                idxs = np.random.choice(np.arange(len(train_loader.dataset)),size=args.num_finetune_complexes,replace=False)
                complex_graphs = [train_loader.dataset[i] for i in idxs]
                finetune_losses, finetune_lddt_losses = finetune_epoch(model, complex_graphs, device, t_to_sigma, args, optimizer, loss_fn, ema_weights)
                print("Epoch {}: Finetune loss {:.4f}  lddt {:.4f}  affinity {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  res_tr {:.4f}   res_rot {:.4f}   res_chi {:.4f}"
                      .format(epoch, finetune_losses['loss'], finetune_losses['lddt_loss'], finetune_losses['affinity_loss'], finetune_losses['tr_loss'], finetune_losses['rot_loss'],
                              finetune_losses['tor_loss'], finetune_losses['res_tr_loss'], finetune_losses['res_rot_loss'], finetune_losses['res_chi_loss']))

                print("Epoch {}: Finetune base loss {:.4f} lddt {:.4f}  affinity {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  res_tr {:.4f}   res_rot {:.4f}   res_chi {:.4f}"
                          .format(epoch, finetune_losses['base_loss'], finetune_losses['lddt_base_loss'], finetune_losses['affinity_base_loss'], finetune_losses['tr_base_loss'], finetune_losses['rot_base_loss'],
                                  finetune_losses['tor_base_loss'], finetune_losses['res_tr_base_loss'], finetune_losses['res_rot_base_loss'], finetune_losses['res_chi_base_loss']))

                print("Epoch {}: Finetune lddt loss {:.4f}  lddt {:.4f}  affinity {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  res_tr {:.4f}   res_rot {:.4f}   res_chi {:.4f}"
                      .format(epoch, finetune_lddt_losses['loss'], finetune_lddt_losses['lddt_loss'], finetune_lddt_losses['affinity_loss'], finetune_lddt_losses['tr_loss'], finetune_lddt_losses['rot_loss'],
                              finetune_lddt_losses['tor_loss'], finetune_lddt_losses['res_tr_loss'], finetune_lddt_losses['res_rot_loss'], finetune_lddt_losses['res_chi_loss']))

                print("Epoch {}: Finetune lddt base loss {:.4f} lddt {:.4f}  affinity {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  res_tr {:.4f}   res_rot {:.4f}   res_chi {:.4f}"
                          .format(epoch, finetune_lddt_losses['base_loss'], finetune_lddt_losses['lddt_base_loss'], finetune_lddt_losses['affinity_base_loss'], finetune_lddt_losses['tr_base_loss'], finetune_lddt_losses['rot_base_loss'],
                                  finetune_lddt_losses['tor_base_loss'], finetune_lddt_losses['res_tr_base_loss'], finetune_lddt_losses['res_rot_base_loss'], finetune_lddt_losses['res_chi_base_loss']))

            ema_weights.store(model.parameters())
            if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
            val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, args.test_sigma_intervals)
            print("Epoch {}: Validation loss {:.4f}  lddt {:.4f}  affinity {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  res_tr {:.4f}   res_rot {:.4f}   res_chi {:.4f}"
                  .format(epoch, val_losses['loss'], val_losses['lddt_loss'], val_losses['affinity_loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss'],
                            val_losses['res_tr_loss'], val_losses['res_rot_loss'], val_losses['res_chi_loss']))

            print("Epoch {}: Validation base loss {:.4f}  lddt {:.4f}  affinity {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  res_tr {:.4f}   res_rot {:.4f}   res_chi {:.4f}"
                      .format(epoch, val_losses['base_loss'], val_losses['lddt_base_loss'], val_losses['affinity_base_loss'], val_losses['tr_base_loss'], val_losses['rot_base_loss'], val_losses['tor_base_loss'],
                                val_losses['res_tr_base_loss'], val_losses['res_rot_base_loss'], val_losses['res_chi_base_loss']))

            if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
                inf_metrics = inference_epoch(model, val_loader.dataset.complex_graphs[:args.num_inference_complexes], device, t_to_sigma, args)
                print("Epoch {}: Val inference lddt_rmse {:.3f} lddt_base_rmse {:.3f} lddt_pearson {:.3f} lddt_spearman {:.3f} affinity_rmse {:.3f} affinity_base_rmse {:.3f} affinity_pearson {:.3f} affinity_spearman {:.3f}"
                      .format(epoch, inf_metrics['lddt_rmse'], inf_metrics['lddt_base_rmse'], inf_metrics['lddt_pearson'], inf_metrics['lddt_spearman'], inf_metrics['affinity_rmse'], inf_metrics['affinity_base_rmse'], inf_metrics['affinity_pearson'], inf_metrics['affinity_spearman']))
                print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}"
                      .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']))

                logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)
        else:
            inf_metrics = inference_epoch(model, val_loader.dataset.complex_graphs[:args.num_inference_complexes], device, t_to_sigma, args)
            print("Epoch {}: Val inference lddt_rmse {:.3f} lddt_base_rmse {:.3f} lddt_pearson {:.3f} lddt_spearman {:.3f} affinity_rmse {:.3f} affinity_base_rmse {:.3f} affinity_pearson {:.3f} affinity_spearman {:.3f}"
                  .format(epoch, inf_metrics['lddt_rmse'], inf_metrics['lddt_base_rmse'], inf_metrics['lddt_pearson'], inf_metrics['lddt_spearman'], inf_metrics['affinity_rmse'], inf_metrics['affinity_base_rmse'], inf_metrics['affinity_pearson'], inf_metrics['affinity_spearman']))
            print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']))
            assert 1==0, 'only inference test'
        if not args.use_ema: ema_weights.copy_to(model.parameters())
        ema_state_dict = copy.deepcopy(model.module.state_dict() if device.type == 'cuda' else model.state_dict())
        ema_weights.restore(model.parameters())

        if args.wandb:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch + 1)

        state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()

        if (epoch + 1) % args.val_inference_freq == 0:
            torch.save(ema_state_dict, os.path.join(run_dir, f'ema_inference_epoch{epoch}_model.pt'))
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))

        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

        if scheduler:
            if args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


def main_function():
    args = parse_train_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)

    model = get_model(args, device, t_to_sigma=t_to_sigma)

    # if len(gpus) > 1:
    #     model = DataParallel(model, device_ids=gpus, output_device=gpus[0])

    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)
    start_epoch = 0
    if args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
            if args.restart_lr is not None: dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(dict['optimizer'])
            model.module.load_state_dict(dict['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(dict['ema_weights'], device=device)
            print("Restarting from epoch", dict['epoch'])
            start_epoch = dict['epoch'] + 1
        except Exception as e:
            print("Exception", e)
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            model.module.load_state_dict(dict, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    if args.wandb:
        wandb.init(
            entity='entity',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'numel': numel})

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    if not args.only_test:
        os.system(f'cp -r datasets {run_dir}')
        os.system(f'cp -r models {run_dir}')
        os.system(f'cp -r utils {run_dir}')
        os.system(f'cp *.py {run_dir}')
        os.system(f'cp *.sh {run_dir}')
        yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
        save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, start_epoch)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
