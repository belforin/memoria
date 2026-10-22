#!/bin/bash
# Metrica 1 de interpretabilidad (belforin/memoria METODOLOGIA_DT_HDT.md
# seccion 7.16): retorno en lazo cerrado segun el largo de la ventana de
# evaluacion K. Envia un job por (dominio, K) con eval_bc_dmc.sbatch /
# eval_bc_procgen.sbatch sin otros cambios (snapshot final, 100 episodios,
# semilla de evaluacion = semilla de entrenamiento).
# K=full = sin EVAL_K (ventana completa: 64 en DMC, 16 en CoinRun), que es la
# evaluacion estandar; para bc_uni/bc_hier ya existe (seccion 7.10).
#
# Uso: AGENTS="bc_uni bc_hier" slurm_bc/submit_context_curve.sh
#      AGENTS="bc_hier223 bc_hier223_enc128" DMC_KS="1 2 4 8 16 32 full" PROCGEN_KS="1 2 4 8 full" \
#          DEPEND=afterok:33205:33206 slurm_bc/submit_context_curve.sh
# Variables: AGENTS, DMC_KS, PROCGEN_KS, EPISODES, DEPEND (--dependency), QOS_ARGS.
set -euo pipefail
cd $HOME/MaskDP_bc
AGENTS=${AGENTS:?ej. "bc_uni bc_hier"}
DMC_KS=${DMC_KS:-1 2 4 8 16 32}
PROCGEN_KS=${PROCGEN_KS:-1 2 4 8}
EPISODES=${EPISODES:-100}
DEP=${DEPEND:+--dependency=$DEPEND}
COMMON="--partition=ialab-high,ialab-low --gres=gpu:1 --exclude=antuco,yodaxico --cpus-per-task=4 --mem=24G ${QOS_ARGS:-}"
TAG=$(echo $AGENTS | tr ' ' '+')

for K in $DMC_KS; do
  [ "$K" = full ] && EK="" || EK=$K
  sbatch --parsable $COMMON $DEP --time=24:00:00 -J bc_ctx_dmc_K${K}_${TAG} \
      --export=ALL,AGENTS="$AGENTS",EVAL_K=$EK,EPISODES=$EPISODES slurm_bc/eval_bc_dmc.sbatch \
      | sed "s/^/DMC K=$K: job /"
done
for K in $PROCGEN_KS; do
  [ "$K" = full ] && EK="" || EK=$K
  sbatch --parsable $COMMON $DEP --time=12:00:00 -J bc_ctx_procgen_K${K}_${TAG} \
      --export=ALL,AGENTS="$AGENTS",EVAL_K=$EK,EPISODES=$EPISODES slurm_bc/eval_bc_procgen.sbatch \
      | sed "s/^/CoinRun K=$K: job /"
done
