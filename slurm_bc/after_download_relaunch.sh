#!/bin/bash
# Espera la descarga de maskdp_data (expert/{cheetah_run,walker_run,quadruped_run}),
# la reintenta si quedó incompleta, y al verificarla apunta DMC_DATA a la copia local,
# libera los jobs retenidos y relanza los que fallaron por el cambio de permisos.
# Requiere HF_TOKEN en el entorno (no se guarda en disco).
set -u
DL_PID=${1:-}
ROOT=/home/belforin/MaskDP_bc/data/maskdp_data
PY=/home/belforin/miniconda3/envs/maskdp-ben/bin/python
SUBS="maskdp_train/cheetah/expert/cheetah_run maskdp_train/walker/expert/walker_run maskdp_train/quadruped/expert/quadruped_run"
EXPECTED=3800
log() { echo "[$(date '+%F %T')] $*"; }

if [ -n "$DL_PID" ]; then
    log "esperando PID $DL_PID"
    while kill -0 "$DL_PID" 2>/dev/null; do sleep 60; done
fi

complete() {
    for s in $SUBS; do
        n=$(find "$ROOT/$s" -type f -name '*.npz' 2>/dev/null | wc -l)
        log "$s: $n/$EXPECTED"
        [ "$n" -ge "$EXPECTED" ] || return 1
    done
}

for try in 1 2 3 4 5; do
    complete && break
    log "incompleto, intento de descarga $try"
    "$PY" -I -c "
from huggingface_hub import snapshot_download
pats=['maskdp_train/cheetah/expert/cheetah_run/*','maskdp_train/walker/expert/walker_run/*','maskdp_train/quadruped/expert/quadruped_run/*']
snapshot_download('fangchenliu/maskdp_data',repo_type='dataset',local_dir='$ROOT',allow_patterns=pats,max_workers=4)
print('DONE')
" || sleep 300
done
if ! complete; then log "ABORTO: descarga incompleta tras 5 intentos"; exit 1; fi

unset HF_TOKEN  # no propagar el token a los jobs (--export=ALL)
cd /home/belforin/MaskDP_bc
sed -i "s|^DMC_DATA=.*|DMC_DATA=$ROOT/maskdp_train|" slurm_bc/paths.sh
log "paths.sh: $(grep ^DMC_DATA slurm_bc/paths.sh)"

scontrol release 33223 33690 33501 && log "liberados 33223 33690 33501"

for spec in bc_hier223:walker_run:3 bc_hier223_enc128:cheetah_run:3 bc_hier223_enc128:cheetah_run:1 bc_hier223:cheetah_run:2; do
    IFS=: read -r AGENT TASK SEED <<< "$spec"
    jid=$(sbatch --parsable --partition=ialab-high,ialab-low --qos=regular --gres=gpu:1 \
        --cpus-per-task=4 --mem=20G --exclude=yodaxico,antuco \
        -J bc_dmc_${AGENT}_${TASK}_s${SEED} \
        --export=ALL,AGENT=$AGENT,TASK=$TASK,SEED=$SEED slurm_bc/train_bc_dmc.sbatch)
    log "lanzado $AGENT $TASK s$SEED -> $jid"
done
log "FIN"
