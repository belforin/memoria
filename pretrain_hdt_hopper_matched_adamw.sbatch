#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_hdt_hopper_matched_adamw
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
    task=hopper_medium_expert \
    obs_dim=11 \
    action_dim=3 \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data/hopper_medium_expert \
    snapshot_dir=$HOME/snapshot/hopper_medium_expert_hdt_adamw \
    hydra.run.dir="$HOME/models/hopper_medium_expert_hdt_adamw/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    num_grad_steps=100010 \
    snapshots=[0,25000,50000,75000,100000] \
    use_wandb=false \
    use_tb=true \
    exp_name=hopper_medium_expert_hdt_matched_adamw
