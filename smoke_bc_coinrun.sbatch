#!/bin/bash
#SBATCH --job-name=maskdp_smoke_bc_coinrun
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment="Smoke test de behaviour cloning sin return-to-go en CoinRun (uni y jerarquico) para medir tiempos antes de entrenar"

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env

cd $HOME/MaskDP_mmodal

# Etapa BC (METODOLOGIA_DT_HDT.md seccion 7): 2000 pasos de cada topologia
# con la config de Benjamin (batch 128, contexto 64, relleno de episodios
# cortos), sin snapshots, solo para medir FPS y proyectar 400k pasos.
for AGENT in bc_uni_coinrun bc_hier_coinrun; do
    HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
        agent=$AGENT \
        task=coinrun \
        replay_buffer_dir=$HOME/MaskDP_mmodal/data \
        pad_short_episodes=true \
        num_grad_steps=2001 \
        log_every_steps=250 \
        "snapshots=[]" \
        snapshot_dir=$HOME/snapshot/smoke_bc \
        hydra.run.dir="$HOME/models/smoke_bc/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
        agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
        use_wandb=false \
        use_tb=true \
        exp_name=smoke_bc_$AGENT
done
