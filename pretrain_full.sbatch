#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_cheetah
#SBATCH --partition=ialab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env

cd $HOME/MaskDP_mmodal

python pretrain.py \
    task=cheetah_run \
    replay_buffer_dir=$HOME/data/maskdp_data/maskdp_train \
    snapshot_dir=$HOME/snapshot/cheetah_full \
    hydra.run.dir="$HOME/models/cheetah_full/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    use_wandb=false \
    use_tb=false \
    exp_name=cheetah_full
