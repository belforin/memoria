#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_hdt_coinrun_rtgfix_s2
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
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

# Segunda semilla de CoinRun rtgfix (METODOLOGIA_DT_HDT.md, seccion 2.10 y
# resumen de proximos pasos, punto 2) -- ver
# pretrain_dt_coinrun_rtgfix_s2.sbatch para el motivo.
HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
    agent=hdt_coinrun \
    task=coinrun \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data \
    snapshot_dir=$HOME/snapshot/coinrun_hdt_rtgfix \
    hydra.run.dir="$HOME/models/coinrun_hdt_rtgfix_s2/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
    seed=2 \
    use_wandb=false \
    use_tb=true \
    exp_name=coinrun_hdt_rtgfix_s2
