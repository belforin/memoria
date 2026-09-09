#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_hdt_walker2d_matched
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
    agent=hdt \
    task=walker2d_medium_expert \
    obs_dim=17 \
    action_dim=6 \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data/walker2d_medium_expert \
    snapshot_dir=$HOME/snapshot/walker2d_medium_expert_hdt \
    hydra.run.dir="$HOME/models/walker2d_medium_expert_hdt/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    num_grad_steps=100010 \
    snapshots=[0,25000,50000,75000,100000] \
    use_wandb=false \
    use_tb=true \
    exp_name=walker2d_medium_expert_hdt_matched
