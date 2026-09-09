#!/bin/bash
#SBATCH --job-name=maskdp_compare
#SBATCH --partition=ialab
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env
cd $HOME/MaskDP_mmodal

python compare_snapshots.py \
    task=cheetah_run \
    replay_buffer_dir=$HOME/data/maskdp_data/maskdp_train \
    snapshot_dir=$HOME/snapshot/cheetah_full \
    replay_buffer_size=5000 \
    replay_buffer_num_workers=4 \
    batch_size=32 \
    use_wandb=false \
    use_tb=false \
    exp_name=snapshot_compare \
    hydra.run.dir="$HOME/models/snapshot_compare/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}"
