export PATH=/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/relax/bin:$PATH
# python='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022'
python='python'
header='test_1qg8_may24'

# # inference
$python run_single_protein_inference.py data/origin-1qg8.pdb data/1qg8_input.csv --samples_per_complex 10 --savings_per_complex 5 --num_workers 20 --inference_steps 20 --header $header --device $1 --python /mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022 --relax_python /mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/relax/bin/python
# # generate movie for the two samples, rank 1 and rank 2, in the first compound-protein pair. --device 
# $python movie_generation.py results/$header/index0_idx_0/ 1+2 --device $1 --python /mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022 --relax_python /mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/relax/bin/python


# header='gpcr'
# # inference
# $python run_single_protein_inference.py data data/gpcr.csv -l --samples_per_complex 40 --savings_per_complex 5 --num_workers 20 --inference_steps 20 --header $header --device $1 --python /mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022 --relax_python /mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/relax/bin/python

# dataFile='data/test_imatinib.csv'
# header='imatinib'
# dataFile='data/cmet_need_relax.csv'
# header='cmet'
# dataFile='data/1qg8_input.csv'
# header='1qg8_shrink'
# python='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022'
# relax_python='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/relax/bin/python'

# $python datasets/esm_embedding_preparation.py --protein_ligand_csv $dataFile --out_file data/prepared_for_esm_$header.fasta

# CUDA_VISIBLE_DEVICES=$1 $python esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_$header.fasta data/esm2_output --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir esm_models

# CUDA_VISIBLE_DEVICES=$1 $python -m inference --ckpt pro_ema_inference_epoch138_model.pt --protein_dynamic --save_visualisation --model_dir workdir/big_score_model_sanyueqi_with_time  --protein_ligand_csv $dataFile --esm_embeddings_path data/esm2_output --out_dir results/$header --inference_steps 20 --samples_per_complex 40 --savings_per_complex 40 --batch_size 5 --actual_steps 20 --no_final_step_noise

# CUDA_VISIBLE_DEVICES=$1 $relax_python -m relax_final --results_path results/$header --samples_per_complex 40

# $python -m save_reverseprocess --results_path results/$header --samples_per_complex 1

# CUDA_VISIBLE_DEVICES=$1 $relax_python -m relax_vis --results_path results/$header --samples_per_complex 1
