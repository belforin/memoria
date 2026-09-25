#!/bin/bash
#SBATCH --job-name=maskdp_eval_coinrun_seeds
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment="Evaluacion CoinRun DT vs HDT (3 semillas) con niveles de Procgen reproducibles"

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate procgen-env

cd $HOME/MaskDP_mmodal
mkdir -p eval_results

# Agrega eval_coinrun.py sobre las 3 semillas rtgfix (jobs 28685/28686 s1,
# 28845-28848 s2/s3) para separar el hallazgo de generalizacion DT vs. HDT
# de varianza de semilla (METODOLOGIA_DT_HDT.md, resumen de proximos pasos,
# punto 12). Sin video (ya existen para semilla 1).
python eval_coinrun_seeds.py --device cpu 2>&1 | tee eval_results/eval_coinrun_seeds_rtgfix.log
