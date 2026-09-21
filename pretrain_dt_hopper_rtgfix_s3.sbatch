#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_dt_hopper_rtgfix_s3
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

# Re-entrenamiento con return-to-go corregido (METODOLOGIA_DT_HDT.md, seccion 2.10):
# rtg del episodio completo y sin descontar (return_to_go=True en pretrain_dt.py).
# Adam simple + normalizacion de obs, semilla 3.
HYDRA_FULL_ERROR=1 python pretrain_dt.py \
    agent=dt \
    task=hopper_medium_expert \
    obs_dim=11 \
    action_dim=3 \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data/hopper_medium_expert \
    snapshot_dir=$HOME/snapshot/hopper_medium_expert_dt_rtgfix \
    hydra.run.dir="$HOME/models/hopper_medium_expert_dt_rtgfix_s3/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    num_grad_steps=100010 \
    snapshots=[0,25000,50000,75000,100000] \
    seed=3 \
    use_wandb=false \
    use_tb=true \
    agent.use_adamw=false \
    exp_name=hopper_medium_expert_dt_rtgfix_s3
