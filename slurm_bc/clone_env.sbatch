#!/bin/bash
#SBATCH --job-name=bc_clone_env
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Copia del entorno conda de Benjamin para correr behaviour cloning en su pipeline sin depender de su instalacion"

# Copia propia de /home/bmancilla/miniconda3/envs/maskdp_procgen (DMC +
# Procgen + torch 1.13 cu113 + hydra), ver METODOLOGIA_DT_HDT.md seccion 7.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -q --clone /home/bmancilla/miniconda3/envs/maskdp_procgen -n maskdp-ben
conda activate maskdp-ben
python -c "import torch, dm_control, procgen, hydra; print('maskdp-ben OK, torch', torch.__version__, torch.version.cuda)"
