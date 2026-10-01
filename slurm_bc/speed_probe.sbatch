#!/bin/bash
#SBATCH --job-name=bc_speed
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Medicion de velocidad de entrenamiento de behaviour cloning por tipo de GPU para repartir corridas"

# 300 pasos de cada configuracion BC (DMC uni/hier, CoinRun uni/hier) en un
# tipo de GPU, para estimar si una corrida completa cabe en 24 h.
# Uso: sbatch --gres=gpu:titan_rtx:1 slurm_bc/speed_probe.sbatch
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${BC_ENV:-maskdp-ben}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
unset DISPLAY
cd $HOME/MaskDP_bc
source slurm_bc/paths.sh
OUT=$HOME/models/bc_speed/$SLURM_JOB_ID
nvidia-smi --query-gpu=name --format=csv,noheader

for AGENT in bc_uni bc_hier; do
    python pretrain.py --config-name pretrain_full \
        agent=$AGENT task=cheetah_run seed=1 +action_repeat=1 \
        replay_buffer_dir="$DMC_DATA" replay_buffer_num_workers=4 \
        num_grad_steps=301 log_every_steps=100 "snapshots=[]" goal_buffer_dir=null \
        snapshot_dir=$OUT/s_dmc_$AGENT hydra.run.dir=$OUT/dmc_$AGENT \
        use_wandb=false use_tb=true exp_name=speed_dmc_$AGENT > /dev/null 2>&1
    echo "dmc_$AGENT fps: $(tail -1 $OUT/dmc_$AGENT/train.csv | cut -d, -f2)"
    python pretrain.py --config-name pretrain_procgen \
        agent=${AGENT}_procgen task=coinrun seed=1 \
        use_live_env=false has_dummy_transition=true train_file_split=all use_raw_replay_dir=true \
        replay_buffer_dir="$PROCGEN_NPZ" replay_buffer_num_workers=4 \
        use_return_eval=false goal_buffer_dir=null obs_type=pixels frame_stack=1 \
        use_bct_eval=false early_stop=false agent.transformer_cfg.pretrained_encoder_path="$IMPALA_CKPT" \
        num_grad_steps=301 log_every_steps=100 "snapshots=[]" \
        snapshot_dir=$OUT/s_procgen_$AGENT hydra.run.dir=$OUT/procgen_$AGENT \
        use_wandb=false use_tb=true exp_name=speed_procgen_$AGENT > /dev/null 2>&1
    echo "procgen_$AGENT fps: $(tail -1 $OUT/procgen_$AGENT/train.csv | cut -d, -f2)"
done
