#!/bin/bash
#SBATCH --job-name=eb-ms
#SBATCH --partition=gre
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/remote_dir/home/qianqianxie/ebisu/logs/%x_%j.out
# CPU-only: download one model from ModelScope (faster than hf-mirror from the WHU cluster) into $EB/models/<name>,
# PAR files in parallel, resumable (curl -C -), retried until every file matches the size ModelScope reports.
# Usage: sbatch --export=ALL,REPO=mistralai/Mistral-Small-24B-Instruct-2501,PAR=12 slurm/fetch_ms.sbatch
set -uo pipefail
source /remote_dir/home/qianqianxie/ebisu/Ebisu_Ben/slurm/env.sh
: "${REPO:?}" "${PAR:=12}"
OUT=$EB/models/$(basename $REPO); mkdir -p $OUT; cd $OUT
LIST=$OUT/.ms_files.tsv
curl -sfL "https://www.modelscope.cn/api/v1/models/$REPO/repo/files?Recursive=true" | $PY_EVAL -c "
import json,sys
for f in json.load(sys.stdin)['Data']['Files']:
    if f['Type']=='blob' and not f['Path'].startswith('original/') and not f['Path'].endswith('.pth'): print(f['Path'], f['Size'], sep='\t')" > $LIST
echo "$(wc -l < $LIST) files, $(awk '{s+=$2} END{printf "%.1f GB", s/1e9}' $LIST)"
for round in $(seq 1 30); do
  awk -F'\t' '{ cmd="stat -c %s \"" $1 "\" 2>/dev/null"; sz=""; cmd | getline sz; close(cmd); if (sz != $2) print $1 }' $LIST > $OUT/.ms_todo
  n=$(wc -l < $OUT/.ms_todo); echo "round $round: $n files left"; [ "$n" -eq 0 ] && break
  cat $OUT/.ms_todo | xargs -P $PAR -I{} sh -c 'mkdir -p "$(dirname "{}")"; curl -sfL -C - --retry 5 --retry-delay 5 -o "{}" "https://www.modelscope.cn/models/'$REPO'/resolve/master/{}" || true'
done
du -sh $OUT
