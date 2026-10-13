#!/bin/bash
#SBATCH --job-name=eb-eval
#SBATCH --partition=gre
#SBATCH --exclude=gpu14,gpu16,gpu17,gpu22,gpu31,gpu33
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/remote_dir/home/qianqianxie/ebisu/logs/%x_%j.out
# Usage: sbatch --gres=gpu:N --export=ALL,MODEL=<hf id or local path>,TASKS=ebisu_oct26|ebisu_jfkrc,MAXLEN=8192 slurm/eval.sbatch
set -euo pipefail
source /remote_dir/home/qianqianxie/ebisu/Ebisu_Ben/slurm/env.sh
: "${MODEL:?}" "${TASKS:=ebisu_oct26}" "${MAXLEN:=8192}" "${EXTRA_ARGS:=}"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "MODEL=$MODEL TASKS=$TASKS MAXLEN=$MAXLEN GPUS=$CUDA_VISIBLE_DEVICES host=$(hostname) commit=$(git -C $EB_CODE rev-parse --short HEAD)"
cd $EB_CODE
$LM_EVAL --model vllm \
  --model_args "pretrained=$MODEL,tensor_parallel_size=$NGPU,gpu_memory_utilization=0.9,max_model_len=$MAXLEN,seed=0${EXTRA_ARGS:+,$EXTRA_ARGS}" \
  --tasks "$TASKS" --batch_size auto --apply_chat_template --log_samples \
  --include_path $EB_CODE/tasks/ebisu_oct26 \
  --output_path $EB_RES/$(basename "$MODEL")
