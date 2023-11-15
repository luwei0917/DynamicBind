1. 环境安装

- Create a new environment

    conda create --name dynamicbind python=3.8

- Activate the environment

    conda activate dynamicbind

- Install

    conda install -c conda-forge cudatoolkit=11.6

    pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

    pip install torch-scatter==2.1.0+pt112cu116 torch-sparse==0.6.16+pt112cu116 torch-cluster==1.6.0+pt112cu116 torch-spline-conv==1.2.1+pt112cu116 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

    pip install torch-geometric==2.2.0

    python -m pip install PyYAML scipy "networkx[default]" biopython e3nn spyrmsd pandas biopandas click

    pip install rdkit-pypi==2022.3.5

    git clone https://github.com/facebookresearch/esm

    cd esm

    pip install -e .

    cd ..

2. 重新训练

# 数据准备

- Structure (全量数据：/gxr/jixian/project/complex_structure_prediction/DiffDynamicPro-correct-sidechain-symmetry/data/PDBBind_af2_aligned)

    每个pdb相关文件单独存储在data/PDBBind_af2_aligned/下，如：data/PDBBind_af2_aligned/1bzy_POP_B，其中应包括晶体结构蛋白、af2结构蛋白和ligand sdf。

    晶体结构蛋白和af2结构蛋白需保证长度一致，并进行了align。命名规则如下：

    {pdb_name}_aligned_to_{uid}.cif  {pdb_name}_ligand.sdf  af2_{pdb_name}_aligned.pdb

- Info (/gxr/jixian/project/complex_structure_prediction/DiffDynamicPro-correct-sidechain-symmetry/data/PDBBind_af2_aligned.csv)

    将所有信息存储为data/PDBBind_af2_aligned.csv， 信息如下：

    crystal_protein_path,protein_path,ligand,name,group,gap_mask,affinity
    data/PDBBind_af2_aligned/10gs_MES_A/10gs_MES_A_aligned_to_P09211.cif,data/PDBBind_af2_aligned/10gs_MES_A/af2_10gs_MES_A_aligned.pdb,data/PDBBind_af2_aligned/10gs_MES_A/10gs_MES_A_ligand.sdf,10gs_MES_A,train,0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,-1.0

- Split (/gxr/jixian/project/complex_structure_prediction/DiffDynamicPro-correct-sidechain-symmetry/data/splits/)

    将数据集划分为train/test，文件中包含{pdb_name} 具体如下：

    ==> data/splits/split_test <==
    1ah0_SBI_A
    1bvr_THT_A
    1bzy_POP_B
    1cl2_PPG_B
    1cnf_ADP_A
    1dgp_FOH_A
    1exv_PLP_B
    1fq0_CIT_A
    1fsa_F6P_B
    1gg1_PGA_C

    ==> data/splits/split_train <==
    10gs_MES_A
    10gs_MES_B
    10mh_SAH_A
    11ba_UPA_A
    11ba_UPA_B
    11bg_U2G_B
    11gs_GSH_A
    11gs_EAA_A
    11gs_MES_A
    11gs_GSH_B

# 预处理

- 激活环境

    conda activate dynamicbind

- 提取序列

    python datasets/pdbbind_lm_embedding_preparation.py

- 提取esm

    CUDA_VISIBLE_DEVICES=$1 python esm/scripts/extract.py esm2_t33_650M_UR50D data/pdbbind_sequences.fasta data/embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 10000

- esm转dict

    python datasets/esm_embeddings_to_pt.py

- 训练

    小模型参数见run.sh

    大模型：
    CUDA_VISIBLE_DEVICES=$1 python -m finetune --split_val data/splits/split_test --run_name big_score_model_sanyueqi_with_time_weight_by_ligand --tr_sigma_min 0.1 --tr_sigma_max 19 --rot_sigma_min 0.03 --rot_sigma_max 1.55 --tor_sigma_min 0.0314 --tor_sigma_max 3.14 --esm_embeddings_path data/esm2_3billion_embeddings.pt --log_dir workdir --lr 1e-3 --batch_size 40 --ns 72 --nv 12 --num_conv_layers 6 --dynamic_max_cross --scheduler plateau --dropout 0.1 --remove_hs --scale_by_sigma --c_alpha_max_neighbors 24 --receptor_radius 15 --num_workers 48 --num_dataloader_workers 1 --pin_memory --cudnn_benchmark --val_inference_freq 1 --num_inference_complexes 500 --use_ema --scheduler_patience 30 --n_epochs 800 

3. inference

- relax 环境安装

  conda remove -n relax --all
  conda create --name relax python=3.8
  conda activate relax
  conda install -c conda-forge libgcc-ng=12.2.0 -y
  conda install -c conda-forge cudatoolkit=11.4.2 openmm=8.0.0
  conda install -c conda-forge ambertools=22.3 compilers=1.5.2 -y
  conda install -c conda-forge  openff-toolkit=0.12.1 widgetsnbextension=3.6.0 -y
  conda install -c conda-forge  openmmforcefields=0.11.2 -y
  conda install -c conda-forge pdbfixer=1.8.1 biopython=1.78 -y

- 在run_single_protein_inference.py中确定模型

- sh run_inference.sh $device

4. inference for finetune

- 准备好需要inference的文件，.csv格式，包含protein_path, ligand, affinity这三列，如下

  protein_path,ligand
  data/PDBBind_af2_aligned/10gs_MES_A/10gs_MES_A_aligned_to_P09211.cif,data/PDBBind_af2_aligned/10gs_MES_A/10gs_MES_A_ligand.sdf

- 修改run_inference_for_finetune.sh中的参数，并运行，然后在out_dir中会产生与指定gpu个数相同的run_inference_for_finetune_job{*}.sh脚本

- 在当前目录，手动执行每个bash {out_dir}/run_inference_for_finetune_job{*}.sh（为方便查看，没有写成nohup一键执行，用多个screening后台并行）

- 使用dynamicbind环境运行: python prepare_for_finetune.py --out_dir {run_inference_for_finetune.sh中的out_dir}, for example: python prepare_for_finetune.py --out_dir finetune_kinase/kinase_finetune_data/

- sh run_finetune.sh {run_inference_for_finetune.sh中的out_dir}, example: bash run_finetune.sh finetune_kinase/kinase_finetune_data/finetune_data_path.csv 

5. screening

- 准备好需要screening的文件，.csv格式，包含protein_path, ligand这两列，如下

  protein_path,ligand
  data/PDBBind_af2_aligned/10gs_MES_A/10gs_MES_A_aligned_to_P09211.cif,data/PDBBind_af2_aligned/10gs_MES_A/10gs_MES_A_ligand.sdf

- 修改run_screening.sh中的参数（指定model，并修改run_single_protein_inference_ft.py），并运行，然后在out_dir中会产生与指定gpu个数相同的run_screening_job{*}.sh脚本（为方便查看，没有写成nohup一键执行，用多个screening后台并行），example： bash results/pred_finetune_kinase_model_2/run_screening_job0.sh

- 使用dynamicbind环境运行: python merge_screening_results.py --out_dir {修改run_screening.sh中的out_dir}, 最终结果为{out_dir}/affinity_prediction.csv
