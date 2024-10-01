#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 10 hour:
#$ -l h_rt=02:00:00
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
conda activate lrd3_new
nvidia-smi
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2593541/cache/lm_eval"
export TOKENIZERS_PARALLELISM=false
export LOGLEVEL=ERROR
export HF_DATASETS_TRUST_REMOTE_CODE=true

MODEL=meta-llama/Llama-2-7b-hf
#MODEL=baffo32/decapoda-research-llama-7B-hf
MODEL_PATH=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3/models--meta-llama--Llama-2-7b-hf
CACHE_DIR=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3

export TRANSFORMERS_CACHE=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3
export HF_HOME=/exports/eddie/scratch/s2593541/lrd/cache_train_llama3

prune_ckpt_path='/exports/eddie/scratch/s2593541/lrd/cache_train_llama3/llama_prune'

RATIO=0.25

#echo "[START] - Start Pruning Model"
#python hf_prune.py --pruning_ratio $RATIO --base_model=$MODEL --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model --cache_dir=$CACHE_DIR
#echo "[FINISH] - Finish Pruning Model"

#prune_model="${prune_ckpt_path}/pytorch_model.bin"
#echo "[START] - Start Tuning"
#python post_training.py --prune_model $prune_model --data_path yahma/alpaca-cleaned --output_dir $prune_ckpt_path/ --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --save_fname $RATIO
#echo "[FINISH] - Finish Prune and Post-Training."
#echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

#echo "You can use the command:"
#echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
#echo "to use the pruned model"

prune_model="${prune_ckpt_path}/pytorch_model.bin"
max_train_samples=3000
EPOCHS=4

if true; then
    for RATIO in 0.14 0.18 0.24
    do
        echo "[START] - Start Pruning Model with RATIO=$RATIO"
        python hf_prune.py --pruning_ratio $RATIO --base_model=$MODEL --device cuda  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model --cache_dir=$CACHE_DIR
        echo "[FINISH] - Finish Pruning Model"

        echo "[START] - Start Tuning for RATIO=$RATIO"
        python post_training.py --prune_model $prune_model --data_path yahma/alpaca-cleaned --output_dir $prune_ckpt_path/ --wandb_project llama_tune --lora_r 8 --num_epochs $EPOCHS --learning_rate 1e-4 --batch_size 64 --save_fname $RATIO --max_train_samples=$max_train_samples --suffix "llama2"
        echo "[FINISH] - Finish Prune and Post-Training for RATIO=$RATIO."
        echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

        echo "You can use the command:"
        echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
        echo "to use the pruned model for RATIO=$RATIO"
    done
fi

cache_dir=/exports/eddie/scratch/s2593541/lrd/cache_train_llama
prune_model="${prune_ckpt_path}/pytorch_model.bin"
for RATIO in 0.14 0.22 0.28
do
    echo "[START] - Start Pruning Model with RATIO=$RATIO"
    python llama3.py --pruning_ratio $RATIO \
                 --device cuda --eval_device cuda \
                 --base_model meta-llama/Meta-Llama-3-8B \
                 --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
                 --block_attention_layer_start 4 --block_attention_layer_end 30 \
                 --save_ckpt_log_name $prune_ckpt_path \
                 --pruner_type taylor --taylor param_first \
                 --max_seq_len 2048 \
                 --cache_dir=$cache_dir \
                 --save_model
     
    echo "[FINISH] - Finish Pruning Model"

    echo "[START] - Start Tuning for RATIO=$RATIO"
   python post_training.py --prune_model $prune_model --suffix "llama3" --data_path yahma/alpaca-cleaned --output_dir $prune_ckpt_path/ --wandb_project llama_tune --lora_r 8 --num_epochs $EPOCHS --learning_rate 1e-4 --batch_size 64 --save_fname $RATIO --max_train_samples=$max_train_samples --suffix="llama3"
     

    echo "[FINISH] - Finish Prune and Post-Training for RATIO=$RATIO."
    echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

    echo "You can use the command:"
    echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
    echo "to use the pruned model for RATIO=$RATIO"
done


