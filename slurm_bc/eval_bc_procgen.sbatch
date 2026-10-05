#!/bin/bash
#SBATCH --job-name=bc_eval_procgen
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Evaluacion closed-loop de behaviour cloning en CoinRun con el eval_bct.py de Benjamin"

# Retorno de BC-uni/BC-hier en CoinRun con el eval_bct.py de Benjamin sin
# modificar (belforin/memoria METODOLOGIA_DT_HDT.md seccion 7.6): epoca 50
# (paso 781.250), 100 episodios por split, muestreo con temperatura 1.0.
# Su eval_bct.py no fija rand_seed en ProcgenEnv: los niveles sorteados
# cambian entre corridas (igual que en sus resultados).
# Variables opcionales: AGENTS, SEEDS, STEP (uno o varios pasos separados por
# espacio), EPISODES, RUN_TAG (sufijo de los directorios de salida).
# eval_bct.py sobrescribe evaluate_bct.csv junto al snapshot: el log es el registro.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${BC_ENV:-maskdp-ben}
export HYDRA_FULL_ERROR=1
cd $HOME/MaskDP_bc
source slurm_bc/paths.sh
STEP=${STEP:-781250}
EPISODES=${EPISODES:-100}
RUN_TAG=${RUN_TAG:-}

for AGENT in ${AGENTS:-bc_uni bc_hier}; do
  for SEED in ${SEEDS:-1 2 3}; do
   for S in $STEP; do
    SNAP=$BC_SNAP/procgen/$AGENT/coinrun/$SEED/snapshot_$S.pt
    if [ ! -f "$SNAP" ]; then echo "[falta] $SNAP"; continue; fi
    if [ "$S" = 781250 ]; then STAG=""; else STAG="_step$S"; fi
    echo "=== $AGENT coinrun s$SEED step $S"
    python eval_bct.py agent=bc_eval agent.path=$SNAP seed=$SEED num_episodes=$EPISODES use_wandb=false \
        hydra.run.dir=$BC_RUNS/eval_procgen/${AGENT}_s${SEED}${STAG}${RUN_TAG} \
        exp_name=eval_${AGENT}_coinrun_s${SEED}${STAG}${RUN_TAG}
   done
  done
done
