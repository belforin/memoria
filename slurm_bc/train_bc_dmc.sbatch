#!/bin/bash
#SBATCH --job-name=bc_train_dmc
#SBATCH --partition=ialab-high
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.out
#SBATCH --error=/home/belforin/MaskDP_bc/slurm_bc/logs/%x_%j.err
#SBATCH --comment="Entrenamiento de behaviour cloning sin return-to-go en DMC dentro del pipeline de Benjamin"

# Una corrida de BC-uni/BC-hier en DMC con el pretrain.py de Benjamin sin
# modificar (belforin/memoria METODOLOGIA_DT_HDT.md seccion 7.6): batch 384,
# contexto 64, 400k pasos, buffer de 2M con 4 workers, snapshot final 400k.
# Uso: sbatch -J bc_dmc_<agent>_<task>_s<seed> \
#        --export=ALL,AGENT=bc_uni,TASK=cheetah_run,SEED=1 slurm_bc/train_bc_dmc.sbatch
set -euo pipefail
: "${AGENT:?bc_uni|bc_hier}" "${TASK:?cheetah_run|walker_run|quadruped_walk}" "${SEED:?}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${BC_ENV:-maskdp-ben}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
unset DISPLAY
cd $HOME/MaskDP_bc
source slurm_bc/paths.sh
NAME=${AGENT}_${TASK}_s${SEED}

python pretrain.py --config-name pretrain_full \
    agent=$AGENT task=$TASK seed=$SEED +action_repeat=1 \
    replay_buffer_dir="$DMC_DATA" replay_buffer_num_workers=4 \
    num_grad_steps=400010 "snapshots=[0,50000,100000,150000,200000,250000,300000,350000,400000]" \
    goal_buffer_dir=null \
    snapshot_dir=$BC_SNAP/dmc/$AGENT hydra.run.dir=$BC_RUNS/dmc/$NAME \
    use_wandb=false use_tb=true exp_name=$NAME
