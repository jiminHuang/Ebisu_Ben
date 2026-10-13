# Shared environment for the Oct-2026 Ebisu rerun on the WHU cluster. Source from every job.
export EB=/remote_dir/home/qianqianxie/ebisu
export EB_CODE=$EB/Ebisu_Ben
export EB_DATA=$EB/data
export EB_RES=$EB/results
export LM_EVAL=/persist_data/home/qianqianxie/conda_envs/finben/bin/lm_eval
export PY_EVAL=/persist_data/home/qianqianxie/conda_envs/finben/bin/python

# all caches on /remote_dir (the /persist_data home disk is nearly full)
export HF_HOME=${HF_HOME:-$EB/.cache/huggingface}
export XDG_CACHE_HOME=$EB/.cache
export VLLM_CACHE_ROOT=$EB/.cache/vllm/${SLURM_JOB_ID:-x}
export TRITON_CACHE_DIR=$EB/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$EB/.cache/torchinductor/${SLURM_JOB_ID:-x}
export TMPDIR=$EB/.cache/tmp
mkdir -p $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMPDIR $EB_RES $EB/logs
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
