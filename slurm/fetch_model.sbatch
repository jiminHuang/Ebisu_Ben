#!/bin/bash
#SBATCH --job-name=eb-dl
#SBATCH --partition=gre
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/remote_dir/home/qianqianxie/ebisu/logs/%x_%j.out
# CPU-only: download one HF model repo through hf-mirror into $EB/models/<name>.
# Usage: sbatch --export=ALL,REPO=meta-llama/Llama-3.3-70B-Instruct slurm/fetch_model.sbatch
set -euo pipefail
source /remote_dir/home/qianqianxie/ebisu/Ebisu_Ben/slurm/env.sh
: "${REPO:?}"
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null || cat ~/.cache/huggingface/token)
OUT=$EB/models/$(basename $REPO); mkdir -p $OUT
HFCLI=/persist_data/home/qianqianxie/conda_envs/finben/bin/huggingface-cli
$HFCLI download $REPO --local-dir $OUT --exclude "original/*" "*.pth" --max-workers 8
du -sh $OUT; ls $OUT
