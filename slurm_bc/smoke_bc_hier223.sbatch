#!/bin/bash
#SBATCH --job-name=bc_smoke_hier223
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Smoke test de BC-hier 2+2+3 (256 y encoders a 128) antes de las corridas completas"

# Smoke test de las variantes BC-hier 2+2+3 (belforin/memoria
# METODOLOGIA_DT_HDT.md seccion 7.15): 1000 pasos de pretrain.py y 2 episodios
# de evaluacion en DMC (cheetah_run) y CoinRun por variante. Mide pasos/s.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${BC_ENV:-maskdp-ben}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
unset DISPLAY
cd $HOME/MaskDP_bc
source slurm_bc/paths.sh
OUT=$HOME/models/bc_smoke/$SLURM_JOB_ID
nvidia-smi --query-gpu=name --format=csv,noheader

for AGENT in ${AGENTS:-bc_hier223 bc_hier223_enc128}; do
    echo "=== DMC cheetah_run $AGENT"
    python pretrain.py --config-name pretrain_full \
        agent=$AGENT task=cheetah_run seed=1 +action_repeat=1 \
        replay_buffer_dir="$DMC_DATA" replay_buffer_num_workers=4 \
        num_grad_steps=1001 log_every_steps=250 "snapshots=[1000]" \
        goal_buffer_dir=null snapshot_dir=$OUT/snap_dmc_$AGENT hydra.run.dir=$OUT/dmc_$AGENT \
        use_wandb=false use_tb=true exp_name=smoke_dmc_$AGENT
    python eval_return.py agent=bc_eval task=cheetah_run seed=1 obs_type=states action_repeat=1 \
        agent.path=$OUT/snap_dmc_$AGENT/cheetah/1/snapshot_1000.pt num_eval_episodes=2 \
        use_wandb=false hydra.run.dir=$OUT/eval_dmc_$AGENT exp_name=smoke_eval_dmc_$AGENT

    echo "=== CoinRun ${AGENT}_procgen"
    python pretrain.py --config-name pretrain_procgen \
        agent=${AGENT}_procgen task=coinrun seed=1 \
        use_live_env=false has_dummy_transition=true train_file_split=all use_raw_replay_dir=true \
        replay_buffer_dir="$PROCGEN_NPZ" replay_buffer_num_workers=4 \
        use_return_eval=false goal_buffer_dir=null obs_type=pixels frame_stack=1 \
        use_bct_eval=false early_stop=false \
        agent.transformer_cfg.pretrained_encoder_path="$IMPALA_CKPT" \
        num_grad_steps=1001 log_every_steps=250 "snapshots=[1000]" \
        snapshot_dir=$OUT/snap_procgen_$AGENT hydra.run.dir=$OUT/procgen_$AGENT \
        use_wandb=false use_tb=true exp_name=smoke_procgen_$AGENT
    python eval_bct.py agent=bc_eval agent.path=$OUT/snap_procgen_$AGENT/coinrun/1/snapshot_1000.pt \
        num_episodes=2 use_wandb=false hydra.run.dir=$OUT/eval_procgen_$AGENT exp_name=smoke_eval_procgen_$AGENT
done
echo "=== smoke OK"
