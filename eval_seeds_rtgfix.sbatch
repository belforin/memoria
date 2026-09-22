#!/bin/bash
#SBATCH --job-name=maskdp_eval_seeds_rtgfix
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

# Tabla dt/hdt x {halfcheetah,hopper,walker2d} agregada sobre las 3 semillas
# del re-entrenamiento rtgfix (METODOLOGIA_DT_HDT.md, seccion 2.10, jobs
# 28429-28442/28708). CPU: eval_dt.py/eval_seeds.py no necesitan GPU.
python eval_seeds.py --device cpu 2>&1 | tee eval_results/eval_seeds_rtgfix.log
