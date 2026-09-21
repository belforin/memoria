#!/bin/bash
#SBATCH --job-name=maskdp_eval_hdt_coinrun
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate procgen-env

cd $HOME/MaskDP_mmodal
mkdir -p eval_results

# procgen-env tiene torch CPU-only, por eso sin GPU. Rollouts closed-loop en
# los 3 splits (train/val/test), 100 episodios c/u -- METODOLOGIA_DT_HDT.md, seccion 3.
python eval_coinrun.py \
    --snapshot $HOME/snapshot/coinrun_hdt/coinrun/1/snapshot_100000.pt \
    --agent hdt \
    --device cpu \
    --video-dir eval_results/videos_coinrun_hdt_100000 2>&1 | tee eval_results/coinrun_hdt_100000.log
