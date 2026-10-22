#!/bin/bash
#SBATCH --job-name=reeval_ben
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Re-evaluacion de los modelos DMC de Benjamin con la alineacion de acciones corregida en su evaluacion de retorno"

# Re-evaluacion de los snapshots DMC finales de Benjamin (reeval_ben/snapshots.tsv)
# con su eval_return.py, original y con la alineacion corregida, mismos
# parametros que sus eval/final/*_return.sh (T_cond=12, T_pred=12,
# replan_freq=1, action_repeat=1, seed de eval = seed de entrenamiento) y 100
# episodios como su tesis. Uso: sbatch --export=ALL,FIRST=1,LAST=9 reeval.sbatch
# (filas del TSV sin contar el encabezado).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/bmancilla/miniconda3/envs/maskdp   # su entorno DMC, solo lectura
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONDONTWRITEBYTECODE=1
unset DISPLAY
HERE=$HOME/MaskDP_bc/reeval_ben
OUT=${OUT:-$HOME/models/reeval_ben}
mkdir -p $OUT
EPISODES=${EPISODES:-100}

tail -n +2 ${TSV:-$HERE/snapshots.tsv} | sed -n "${FIRST},${LAST}p" | while IFS=$'\t' read -r ARCH COND DOM TASK SEED REPO SNAP; do
  EXTRA=""
  [[ "$REPO" == *MaskDP_paper ]] && EXTRA="obs_type=states"   # como su maskdp_return.sh
  for MODE in original fixed; do
    TAG=${ARCH}_${COND}_${DOM}_s${SEED}_${MODE}
    LINE=$(python $HERE/eval_return_aligned.py --repo $REPO --mode $MODE -- \
        task=$TASK seed=$SEED action_repeat=1 $EXTRA \
        "agent.path='$SNAP'" agent.T_cond=12 agent.T_pred=12 agent.replan_freq=1 \
        num_eval_episodes=$EPISODES use_wandb=false use_tb=false \
        hydra.run.dir=$OUT/runs/$TAG exp_name=$TAG 2>&1 | tee $OUT/runs_$TAG.log | grep -E "^  return=" | tail -1)
    echo -e "$ARCH\t$COND\t$DOM\t$SEED\t$MODE\t$LINE" | tee -a $OUT/results.tsv
  done
done
