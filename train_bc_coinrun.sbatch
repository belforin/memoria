#!/bin/bash
#SBATCH --job-name=maskdp_train_bc_coinrun
#SBATCH --partition=ialab-high
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment="Entrenamiento de behaviour cloning sin return-to-go en CoinRun (uni/jerarquico) comparable con MaskDP de Benjamin"

# Etapa BC (METODOLOGIA_DT_HDT.md seccion 7): una corrida de 400k pasos con
# la config de Benjamin sobre los embeddings precalculados del IMPALA
# (data_emb, seccion 7.4). Uso:
#   sbatch -J maskdp_train_bc_uni_s1 --export=ALL,AGENT=bc_uni_coinrun,SEED=1 train_bc_coinrun.sbatch
set -euo pipefail
: "${AGENT:?falta AGENT (bc_uni_coinrun|bc_hier_coinrun)}"
: "${SEED:?falta SEED}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env

cd $HOME/MaskDP_mmodal

NAME=coinrun_${AGENT%_coinrun}
HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
    agent=$AGENT \
    task=coinrun \
    seed=$SEED \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data_emb \
    pad_short_episodes=true \
    num_grad_steps=400010 \
    "snapshots=[0,5000,10000,25000,50000,75000,100000,150000,200000,250000,300000,350000,400000]" \
    snapshot_dir=$HOME/snapshot/$NAME \
    hydra.run.dir="$HOME/models/$NAME/s$SEED/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
    use_wandb=false \
    use_tb=true \
    exp_name=${NAME}_s$SEED
