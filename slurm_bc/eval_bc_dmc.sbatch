#!/bin/bash
#SBATCH --job-name=bc_eval_dmc
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Evaluacion de retorno de behaviour cloning en DMC con el eval_return.py de Benjamin"

# Retorno en lazo cerrado de BC-uni/BC-hier en DMC con el eval_return.py de
# Benjamin sin modificar (belforin/memoria METODOLOGIA_DT_HDT.md seccion 7.6):
# snapshot 400k, 100 episodios, action_repeat=1, seed de evaluacion = seed de
# entrenamiento (como sus scripts eval/final/*_return.sh).
# Variables opcionales: AGENTS, TASKS, SEEDS, STEP, EPISODES, EVAL_K (largo de
# la ventana de evaluacion; vacio = traj_length del snapshot, 64 en DMC).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${BC_ENV:-maskdp-ben}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
unset DISPLAY
cd $HOME/MaskDP_bc
source slurm_bc/paths.sh
STEP=${STEP:-400000}
EPISODES=${EPISODES:-100}
EVAL_K=${EVAL_K:-}
if [ -n "$EVAL_K" ]; then K_ARGS="agent.K=$EVAL_K"; K_TAG="_K$EVAL_K"; else K_ARGS=""; K_TAG=""; fi
echo "EVAL_K=${EVAL_K:-traj_length}"

for AGENT in ${AGENTS:-bc_uni bc_hier}; do
  for TASK in ${TASKS:-cheetah_run walker_run quadruped_walk}; do
    DOMAIN=${TASK%%_*}
    for SEED in ${SEEDS:-1 2 3}; do
      SNAP=$BC_SNAP/dmc/$AGENT/$DOMAIN/$SEED/snapshot_$STEP.pt
      if [ ! -f "$SNAP" ]; then echo "[falta] $SNAP"; continue; fi
      echo "=== $AGENT $TASK s$SEED"
      python eval_return.py agent=bc_eval task=$TASK seed=$SEED obs_type=states action_repeat=1 \
          agent.path=$SNAP $K_ARGS num_eval_episodes=$EPISODES use_wandb=false \
          hydra.run.dir=$BC_RUNS/eval_dmc/${AGENT}_${TASK}_s${SEED}${K_TAG} \
          exp_name=eval_${AGENT}_${TASK}_s${SEED}${K_TAG} 2>&1 | grep -E "return=|K=|Error|Traceback"
    done
  done
done
