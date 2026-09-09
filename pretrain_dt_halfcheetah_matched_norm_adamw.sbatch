#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_dt_halfcheetah_matched_norm_adamw
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

# Config matcheada a la Tabla 9 / experiment.py del repo oficial de
# Decision Transformer (dataset medium-expert-v2, unico con numero
# publicado citable). Ver METODOLOGIA_DT_HDT.md, seccion 2.5.
HYDRA_FULL_ERROR=1 python pretrain_dt.py \
    agent=dt \
    task=halfcheetah_medium_expert \
    obs_dim=17 \
    action_dim=6 \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data/halfcheetah_medium_expert \
    snapshot_dir=$HOME/snapshot/halfcheetah_medium_expert_dt_norm_adamw \
    hydra.run.dir="$HOME/models/halfcheetah_medium_expert_dt_norm_adamw/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    num_grad_steps=100010 \
    snapshots=[0,25000,50000,75000,100000] \
    use_wandb=false \
    use_tb=true \
    agent.use_adamw=true \
    exp_name=halfcheetah_medium_expert_dt_matched_norm_adamw
