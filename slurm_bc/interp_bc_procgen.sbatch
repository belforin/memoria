#!/bin/bash
#SBATCH --job-name=bc_interp_procgen
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Interpretabilidad (AttAttr y SARFA) de behaviour cloning en CoinRun con el protocolo de la tesis de Benjamin"

# Interpretabilidad de BC-uni/BC-hier en CoinRun con el protocolo de la tesis
# de Benjamin (seccion 4.3; belforin/memoria METODOLOGIA_DT_HDT.md seccion 7.11):
#  1. grabar 50 niveles de test (aleatorios desde 250, --level-seed 1, los mismos
#     para todos los modelos), un episodio por nivel, muestreo con temperatura 1.0
#     (analysis/eval_bct_levels.py, igual que sus grabaciones);
#  2. AttAttr en todos los estados grabados, m=20, objetivo = accion argmax
#     (analysis/bc_attn_attr.py);
#  3. SARFA espacial y temporal sobre 20 estados por episodio de los ya
#     atribuidos (su eval_sarfa.py, phase=evaluate).
# Variables: AGENTS, SEEDS, N_LEVELS (default 50), STEP (default 781250),
# RUN_TAG (sufijo del directorio de salida), STAGES (default "record attattr sarfa").
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${BC_ENV:-maskdp-ben}
export HYDRA_FULL_ERROR=1
cd $HOME/MaskDP_bc
source slurm_bc/paths.sh
STEP=${STEP:-781250}
N_LEVELS=${N_LEVELS:-50}
RUN_TAG=${RUN_TAG:-}
STAGES=${STAGES:-record attattr sarfa}
ROOT=$BC_RUNS/interp_procgen$RUN_TAG
echo "git: $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) archivos modificados"

for AGENT in ${AGENTS:-bc_uni bc_hier}; do
  ARCH=${AGENT#bc_}
  for SEED in ${SEEDS:-1 2 3}; do
    SNAP=$BC_SNAP/procgen/$AGENT/coinrun/$SEED/snapshot_$STEP.pt
    if [ ! -f "$SNAP" ]; then echo "[falta] $SNAP"; continue; fi
    NAME=${AGENT}_s$SEED
    OUT=$ROOT/$NAME
    mkdir -p $OUT
    echo "=== $NAME ($SNAP)"

    if [[ " $STAGES " == *" record "* ]]; then
      date; echo "--- record"
      python -u analysis/eval_bct_levels.py --path $SNAP --env-name coinrun --distribution-mode easy \
          --start-level 250 --end-level 0 --n-levels $N_LEVELS --level-seed 1 \
          --sample --temperature 1.0 --seed $SEED --device cuda \
          --out $OUT/levels.csv --record-dir $OUT/seqs --level-category test --arch $AGENT || continue
    fi
    if [[ " $STAGES " == *" attattr "* ]]; then
      date; echo "--- attattr"
      python -u analysis/bc_attn_attr.py --path $SNAP --model-name $NAME \
          --states-dir $OUT/seqs/test --out-dir $OUT/attattr --m 20 --device cuda || continue
    fi
    if [[ " $STAGES " == *" sarfa "* ]]; then
      date; echo "--- sarfa"
      MODELS="[{name: $NAME, arch: $ARCH, seed: $SEED, path: $SNAP, states_dir: $OUT/seqs/test, attattr_index: $OUT/attattr/index.csv}]"
      python -u eval_sarfa.py models="$MODELS" expected_K=16 output_dir=$ROOT/sarfa device=cuda \
          use_wandb=false exp_name=sarfa_$NAME hydra.run.dir=$OUT/sarfa_hydra
    fi
    date
  done
done
