#!/bin/bash
#SBATCH --job-name=eb-net
#SBATCH --partition=gre
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/remote_dir/home/qianqianxie/ebisu/logs/%x_%j.out
# CPU-only: is the download bottleneck the network or the NFS write path?
# Same 4 GB shard, 30 s per test: (a) to /dev/null, (b) to local /tmp, (c) to NFS, then 8 parallel ranges to /dev/null.
U=https://hf-mirror.com/Qwen/Qwen2.5-32B-Instruct/resolve/main/model-00002-of-00017.safetensors
NFS=/remote_dir/home/qianqianxie/ebisu/.cache/tmp/netprobe_$SLURM_JOB_ID; LOC=/tmp/netprobe_$SLURM_JOB_ID
echo "host $(hostname)"; df -h /tmp | tail -1
t () { curl -sL --max-time 30 -o "$2" -w "$1: %{speed_download} B/s (%{size_download} B)\n" "$U"; }
t "single -> /dev/null" /dev/null
t "single -> local /tmp" $LOC
t "single -> NFS" $NFS
rm -f $LOC $NFS
echo "8 parallel ranges -> /dev/null:"
for i in $(seq 0 7); do s=$((i*400000000)); curl -sL --max-time 30 -r $s-$((s+399999999)) -o /dev/null -w "%{speed_download}\n" "$U" & done | awk '{s+=$1; print "  conn " NR ": " $1} END{print "  aggregate: " s " B/s"}'
echo "dd write test (2 GB) local vs NFS:"
dd if=/dev/zero of=$LOC bs=8M count=256 oflag=direct 2>&1 | tail -1; rm -f $LOC
dd if=/dev/zero of=$NFS bs=8M count=256 conv=fsync 2>&1 | tail -1; rm -f $NFS
