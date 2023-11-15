python='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022'
relax_python='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/relax/bin/python'
# please manually add model options in run_single_protein_inference_ft.py to use the intended model (probably your fine-tuned model).
# for example: add
# if args.model == 2:
#     model_workdir = f"{script_folder}/workdir/finetune_kinase"
#     ckpt = "ema_inference_epoch2_model.pt"
# set --model option 2.
model=2
$python prepare_for_screening.py --protein_ligand_csv ./finetune_kinase/kinase_selecetd.csv --model $model --gpu_id "0" --out_dir ./results/pred_finetune_kinase_model_$model --python $python --relax_python $relax_python
