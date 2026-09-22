#!/bin/bash
#SBATCH --job-name=maskdp_pretrain_dt_coinrun_rtgfix_s2
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
# resumen de proximos pasos, punto 2): la corrida de semilla 1 (28685) sugiere
# que DT sobreajusta mas a los niveles de entrenamiento que HDT tras el fix
# del rtg, pero con una sola semilla no se puede separar de varianza de
# entrenamiento. Mismo protocolo que pretrain_dt_coinrun_rtgfix.sbatch, solo
# cambia seed/snapshot_dir/exp_name (snapshot_dir compartido -> queda en
# .../coinrun_dt_rtgfix/coinrun/2/, no pisa la semilla 1).
HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
    agent=dt_coinrun \
    task=coinrun \
    replay_buffer_dir=$HOME/MaskDP_mmodal/data \
    snapshot_dir=$HOME/snapshot/coinrun_dt_rtgfix \
    hydra.run.dir="$HOME/models/coinrun_dt_rtgfix_s2/\${now:%Y.%m.%d}/\${now:%H%M%S}_\${agent.name}" \
    agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
    seed=2 \
    use_wandb=false \
    use_tb=true \
    exp_name=coinrun_dt_rtgfix_s2
