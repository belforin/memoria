#!/bin/bash
#SBATCH --job-name=maskdp_best_seed_videos_rtgfix
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dt-env

cd $HOME/MaskDP_mmodal
mkdir -p eval_results

# Para cada (agente, tarea) evalua las 3 semillas rtgfix, elige la de mejor
# score D4RL normalizado y graba en video el primer episodio de esa semilla
# ganadora (METODOLOGIA_DT_HDT.md, seccion 2.10).
python best_seed_videos.py --device cpu 2>&1 | tee eval_results/best_seed_videos_rtgfix.log
