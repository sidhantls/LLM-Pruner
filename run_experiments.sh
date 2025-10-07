# llama-2-7b

MODEL=meta-llama/Llama-2-7b-hf
CACHE_DIR=cache_train_llama2/llm_pruner/

prune_ckpt_path='llama_prune'
tune_ckpt_path="${CACHE_DIR}"/tuned_model/

rm -r $CACHE_DIR

mkdir -p "$CACHE_DIR"
mkdir -p "$tune_ckpt_path"

for RATIO in 0.14 0.18 0.25
do
  # Pruning the model
  python hf_prune.py --base_model=$MODEL --pruning_ratio=$RATIO --device cpu --eval_device cuda --block_wise \
  --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 \
  --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --taylor param_first --save_model \
  --cache_dir=$CACHE_DIR

  echo "Load from: ${CACHE_DIR}${prune_ckpt_path}/pytorch_model.bin"

  python post_training.py --base_model=$MODEL --prune_model "${CACHE_DIR}${prune_ckpt_path}"/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir $tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --suffix=$RATIO
  rm -r $CACHE_DIR

done

