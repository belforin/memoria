#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_hdt_coinrun_rtgfix
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

# Re-entrenamiento de CoinRun con return-to-go corregido (METODOLOGIA_DT_HDT.md,
# seccion 2.10/2.11 -- pretrain_coinrun.py ya pasa return_to_go=True, asi que
# esta corrida usa el fix sin cambios de codigo, solo snapshot_dir/exp_name
# nuevos para no pisar los snapshots viejos (jobs 27108, rtg defectuoso).
HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
    agent=hdt_coinrun \
    task=coinrun \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data \
    snapshot_dir=$HOME/snapshot/coinrun_hdt_rtgfix \
    hydra.run.dir="$HOME/models/coinrun_hdt_rtgfix/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
    use_wandb=false \
    use_tb=true \
    exp_name=coinrun_hdt_rtgfix
