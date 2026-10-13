#!/bin/bash
#SBATCH --job-name=eb-probe
#SBATCH --partition=gre
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/remote_dir/home/qianqianxie/ebisu/logs/%x_%j.out
# CPU-only: report env versions, data presence and which target models are already cached anywhere under /remote_dir.
source /remote_dir/home/qianqianxie/ebisu/Ebisu_Ben/slurm/env.sh
$PY_EVAL -c "import lm_eval,vllm,transformers,torch;print('lm_eval',lm_eval.__version__,'vllm',vllm.__version__,'transformers',transformers.__version__,'torch',torch.__version__)"
wc -l $EB_DATA/*/*.jsonl
df -h /remote_dir | tail -1
echo "== cached hub dirs"
for c in /remote_dir/home/qianqianxie/*/.cache/huggingface/hub /remote_dir/home/qianqianxie/.cache/huggingface/hub /remote_dir/home/qianqianxie/models; do
  [ -d "$c" ] && ls -d "$c"/* 2>/dev/null | grep -iE "llama|qwen|mistral|swallow|gemma|nekomata|finma|stablelm|deepseek" | while read d; do echo "$(du -sh "$d" 2>/dev/null | cut -f1) $d"; done
done
