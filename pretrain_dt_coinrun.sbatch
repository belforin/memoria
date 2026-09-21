#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_dt_coinrun
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
# IMPALA congelado (pretrained_encoder_path en agent/dt_coinrun.yaml),
# acciones discretas (Procgen, 15 acciones).
# pretrained_encoder_path se pasa absoluto por CLI: hydra cambia el cwd a
# hydra.run.dir antes de instanciar el agente, asi que la ruta relativa por
# default en agent/dt_coinrun.yaml no resuelve salvo que se corra desde la
# raiz del repo sin hydra.run.dir override (como los smoke tests).
HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
    agent=dt_coinrun \
    task=coinrun \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data \
    snapshot_dir=$HOME/snapshot/coinrun_dt \
    hydra.run.dir="$HOME/models/coinrun_dt/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
    use_wandb=false \
    use_tb=true \
    exp_name=coinrun_dt
