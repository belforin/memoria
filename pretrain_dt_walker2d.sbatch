#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_dt_walker2d
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --nodelist=hydra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env

cd $HOME/MaskDP_mmodal

HYDRA_FULL_ERROR=1 python pretrain_dt.py \
    agent=dt \
    task=walker2d_expert \
    obs_dim=17 \
    action_dim=6 \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data/walker2d_expert \
    snapshot_dir=$HOME/snapshot/walker2d_dt \
    hydra.run.dir="$HOME/models/walker2d_dt/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    use_wandb=false \
    use_tb=true \
    exp_name=walker2d_dt_full
