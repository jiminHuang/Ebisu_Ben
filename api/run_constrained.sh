#!/bin/bash
# Schema-constrained re-evaluation of the API models (one pass, then retry gateway errors at low concurrency).
cd "$(dirname "$0")/.."
export GATEWAY_KEY=$(grep -E '^HERC_GATEWAY_KEY=' /Users/user/claude-pro/TFAI/herculean/.env | cut -d= -f2- | tr -d '"'"'")
O=/Users/user/claude-pro/TFAI/EMNLP2026/results_api_constrained; mkdir -p $O
H=/private/tmp/claude-502/-Users-user-claude-pro-TFAI-EMNLP2026/ed77627e-6b7b-420f-8565-4f2da3796c30/scratchpad/hf
PY=/Users/user/claude-pro/TFAI/herculean/.venv/bin/python
one () { $PY api/run_api.py --constrained "$@" --data $H --out $O --conc 16; $PY api/run_api.py --constrained --retry-errors "$@" --data $H --out $O --conc 6; }
for m in gpt-4o Llama-3.3-70B-Instruct Llama-4-Maverick-17B-128E-Instruct-FP8 DeepSeek-V3.2; do one --model $m > $O/$m.log 2>&1 & done
one --model gpt-5 --max-tokens 16384 --tag 16k > $O/gpt-5_16k.log 2>&1 &
wait
tail -n1 $O/*.log
