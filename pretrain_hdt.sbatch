#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_hdt
#SBATCH --partition=ialab-low-unlimit
#SBATCH --qos=debug
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env

cd $HOME/MaskDP_mmodal

HYDRA_FULL_ERROR=1 python pretrain_dt.py \
    agent=hdt \
    task=halfcheetah_expert \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data/halfcheetah_expert \
    snapshot_dir=$HOME/snapshot/halfcheetah_hdt \
    hydra.run.dir="$HOME/models/halfcheetah_hdt/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    num_grad_steps=200 \
    log_every_steps=10 \
    eval_every_steps=100 \
    num_eval_episodes=2 \
    use_wandb=false \
    use_tb=true \
    exp_name=hdt_smoketest
