#!/bin/bash
#SBATCH --job-name=bc_train_procgen
#SBATCH --partition=ialab-high
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Entrenamiento de behaviour cloning sin return-to-go en CoinRun dentro del pipeline de Benjamin"

# Una corrida de BC-uni/BC-hier en CoinRun con el pretrain.py de Benjamin sin
# modificar (belforin/memoria METODOLOGIA_DT_HDT.md seccion 7.6): batch 64,
# contexto 16, 50 epocas (1 epoca = 1M/64 = 15.625 pasos), snapshot cada 5
# epocas como su evaluacion BCT; se reporta la epoca 50 (paso 781.250).
# Uso: sbatch -J bc_procgen_<agent>_s<seed> \
#        --export=ALL,AGENT=bc_uni,SEED=1 slurm_bc/train_bc_procgen.sbatch
set -euo pipefail
: "${AGENT:?bc_uni|bc_hier}" "${SEED:?}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${BC_ENV:-maskdp-ben}
export HYDRA_FULL_ERROR=1
cd $HOME/MaskDP_bc
source slurm_bc/paths.sh
NAME=${AGENT}_coinrun_s${SEED}
SNAPS="[$(python -c 'print(",".join(str(e * 15625) for e in range(0, 51, 5)))')]"

python pretrain.py --config-name pretrain_procgen \
    agent=${AGENT}_procgen task=coinrun seed=$SEED \
    use_live_env=false has_dummy_transition=true train_file_split=all use_raw_replay_dir=true \
    replay_buffer_dir="$PROCGEN_NPZ" replay_buffer_num_workers=4 \
    use_return_eval=false goal_buffer_dir=null obs_type=pixels frame_stack=1 \
    use_bct_eval=false early_stop=false \
    agent.transformer_cfg.pretrained_encoder_path="$IMPALA_CKPT" \
    num_grad_steps=781251 "snapshots=$SNAPS" \
    snapshot_dir=$BC_SNAP/procgen/$AGENT hydra.run.dir=$BC_RUNS/procgen/$NAME \
    use_wandb=false use_tb=true exp_name=$NAME
