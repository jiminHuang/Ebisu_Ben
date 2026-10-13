#!/bin/bash
#SBATCH --job-name=eb-data
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/remote_dir/home/qianqianxie/ebisu/logs/%x_%j.out
# CPU-only: pull the three dataset files through hf-mirror (needs an HF token with TheFinAI access in $HF_HOME/token or ~/.cache/huggingface/token).
source /remote_dir/home/qianqianxie/ebisu/Ebisu_Ben/slurm/env.sh
TOKEN=$(cat $HF_HOME/token 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null || true)
[ -n "$TOKEN" ] && echo "token found" || echo "NO HF TOKEN"
get () { mkdir -p $EB_DATA/$1; curl -sfL -H "Authorization: Bearer $TOKEN" -o $EB_DATA/$1/$2 "$HF_ENDPOINT/datasets/TheFinAI/$1/resolve/main/$2" && echo "ok $1/$2 $(wc -l < $EB_DATA/$1/$2)" || echo "FAIL $1/$2"; }
get jp-icr JF-ICR_test_hf_ready_303.jsonl
get jp-te jf_te.jsonl
get JF-KRC-3 jf_krc_merged_release_compact_v2b.jsonl
md5sum $EB_DATA/*/*.jsonl
