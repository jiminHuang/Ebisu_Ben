#!/bin/bash
#SBATCH --job-name=eb-find
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/remote_dir/home/qianqianxie/ebisu/logs/%x_%j.out
# CPU-only: look for existing copies of the target models on /remote_dir (config.json dirs, shallow), and test ModelScope speed.
pat='Swallow-70B|Mistral-Small-24B|Qwen3-32B|Qwen3-14B|Qwen2.5-32B-Instruct|gemma-3n-E4B|finma-7b|TinySwallow|japanese-stablelm-instruct-beta-7b|nekomata-14b'
timeout 600 find /remote_dir -maxdepth 6 -name config.json -path "*" 2>/dev/null | grep -E "$pat" | grep -v '/ebisu/models/' | while read f; do d=$(dirname $f); n=$(ls $d/*.safetensors $d/*.bin 2>/dev/null | wc -l); echo "$n shards  $d"; done
echo "== modelscope speed"
curl -sL -o /dev/null -w "%{speed_download} B/s %{http_code}\n" --max-time 20 "https://www.modelscope.cn/models/Qwen/Qwen3-32B/resolve/master/model-00001-of-00017.safetensors"
curl -sL -o /dev/null -w "%{speed_download} B/s %{http_code}\n" --max-time 20 "https://hf-mirror.com/Qwen/Qwen3-32B/resolve/main/model-00001-of-00017.safetensors"
