#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_hdt_coinrun
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

# Etapa 2 (visual, CoinRun) -- METODOLOGIA_DT_HDT.md, seccion 3. Encoder
# IMPALA congelado (pretrained_encoder_path en agent/hdt_coinrun.yaml),
# acciones discretas (Procgen, 15 acciones).
# pretrained_encoder_path se pasa absoluto por CLI, ver nota en
# pretrain_dt_coinrun.sbatch.
HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
    agent=hdt_coinrun \
    task=coinrun \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data \
    snapshot_dir=$HOME/snapshot/coinrun_hdt \
    hydra.run.dir="$HOME/models/coinrun_hdt/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
    use_wandb=false \
    use_tb=true \
    exp_name=coinrun_hdt
