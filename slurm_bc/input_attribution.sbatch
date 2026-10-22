#!/bin/bash
#SBATCH --job-name=bc_input_attr
#SBATCH --partition=ialab-high,ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --exclude=antuco,yodaxico
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Atribucion de entrada (Integrated Gradients + AOPC) de behaviour cloning"

# Metrica 4 de interpretabilidad (belforin/memoria METODOLOGIA_DT_HDT.md
# seccion 7.16): analysis/input_attribution.py sobre los snapshots finales de
# BC en DMC (400k) y CoinRun (epoca 50). Mismas ventanas (--seed 0) para
# todos los modelos de una tarea. Salta los snapshots que no existan.
# Variables opcionales: AGENTS, TASKS (cheetah_run walker_run quadruped_walk
# coinrun), SEEDS, N_WINDOWS, N_DONORS, M (pasos de IG).
# Con la QOS regular llena: sbatch --partition=ialab-high-unlimit,ialab-low-unlimit --qos=debug --time=01:00:00 ...
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${BC_ENV:-maskdp-ben}
cd $HOME/MaskDP_bc
source slurm_bc/paths.sh
OUT=$BC_RUNS/input_attribution
N_WINDOWS=${N_WINDOWS:-512}
N_DONORS=${N_DONORS:-4}
M=${M:-128}

for TASK in ${TASKS:-cheetah_run walker_run quadruped_walk coinrun}; do
  if [ "$TASK" = coinrun ]; then
    DATA=$PROCGEN_NPZ
  else
    DOMAIN=${TASK%%_*}
    # datos de la tarea evaluada (en quadruped su buffer entreno con quadruped_run, seccion 7.6)
    DATA=$DMC_DATA/$DOMAIN/expert/$TASK/train
  fi
  for AGENT in ${AGENTS:-bc_uni bc_hier bc_hier223 bc_hier223_enc128}; do
    for SEED in ${SEEDS:-1 2 3}; do
      if [ "$TASK" = coinrun ]; then
        SNAP=$BC_SNAP/procgen/$AGENT/coinrun/$SEED/snapshot_781250.pt
      else
        SNAP=$BC_SNAP/dmc/$AGENT/$DOMAIN/$SEED/snapshot_400000.pt
      fi
      if [ ! -f "$SNAP" ]; then echo "[falta] $SNAP"; continue; fi
      if [ -f "$OUT/${AGENT}_${TASK}_s${SEED}.json" ]; then echo "[ya existe] ${AGENT}_${TASK}_s${SEED}"; continue; fi
      echo "=== $AGENT $TASK s$SEED"
      python analysis/input_attribution.py --snapshot $SNAP --data_dir $DATA \
          --out $OUT/${AGENT}_${TASK}_s${SEED} --n_windows $N_WINDOWS --n_donors $N_DONORS --m $M --seed 0
    done
  done
done
echo "=== fin"
