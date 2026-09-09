#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_hdt_full
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
    task=halfcheetah_expert \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data/halfcheetah_expert \
    snapshot_dir=$HOME/snapshot/halfcheetah_hdt \
    hydra.run.dir="$HOME/models/halfcheetah_hdt/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    use_wandb=false \
    use_tb=true \
    exp_name=halfcheetah_hdt_full
