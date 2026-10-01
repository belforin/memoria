#!/bin/bash
#SBATCH --job-name=maskdp_precompute_emb_coinrun
#SBATCH --partition=ialab-high
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment="Precalculo de embeddings del encoder visual congelado de CoinRun para acelerar el entrenamiento de behaviour cloning"

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env

cd $HOME/MaskDP_mmodal

# Etapa BC (METODOLOGIA_DT_HDT.md seccion 7.4): 1) embeddings del IMPALA
# congelado para todo data/coinrun, 2) verificacion contra el camino con
# pixeles, 3) smoke de 2000 pasos por topologia sobre los embeddings para
# medir FPS en A40.
python precompute_coinrun_embeddings.py --src data/coinrun --dst data_emb/coinrun
python test_precomputed_embeddings.py --device cuda

for AGENT in bc_uni_coinrun bc_hier_coinrun; do
    HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
        agent=$AGENT \
        task=coinrun \
        replay_buffer_dir=$HOME/MaskDP_mmodal/data_emb \
        pad_short_episodes=true \
        num_grad_steps=2001 \
        log_every_steps=250 \
        "snapshots=[]" \
        snapshot_dir=$HOME/snapshot/smoke_bc_emb \
        hydra.run.dir="$HOME/models/smoke_bc_emb/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
        agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
        use_wandb=false \
        use_tb=true \
        exp_name=smoke_bc_emb_$AGENT
done
