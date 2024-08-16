#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 10 hour:
#$ -l h_rt=03:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU in the gpu queue:
#$ -q gpu 
#$ -pe gpu-a100 1
#
# Request 4 GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=175G

# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda
#module load cuda/12.1.1
module load anaconda 
conda config --add envs_dirs /exports/eddie/scratch/s2593541/anaconda/envs
conda config --add pkgs_dirs /exports/eddie/scratch/s2593541/anaconda/pkgs
conda activate lrd3
nvidia-smi
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2593541/cache/lm_eval"
export TOKENIZERS_PARALLELISM=false
export LOGLEVEL=ERROR

MODEL=meta-llama/Llama-2-7b-hf
MODEL_PATH=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3/models--meta-llama--Llama-2-7b-hf

python hf_prune.py --pruning_ratio 0.20 \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
      --block_attention_layer_start 4 --block_attention_layer_end 30 \
      --pruner_type taylor \
      --test_after_train \
      --device cuda \
      --eval_device cuda \
      --save_ckpt_log_name llama_prune 
      --base_model $MODEL_PATH
