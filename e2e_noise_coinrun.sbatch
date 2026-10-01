#!/bin/bash
#SBATCH --job-name=maskdp_e2e_noise
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment="Control de sensibilidad numerica del entrenamiento de behaviour cloning en CoinRun (ruido del tamano del redondeo en float32)"

# Etapa BC (METODOLOGIA_DT_HDT.md seccion 7.4): en una misma GPU, 300 pasos
# de BC-uni sobre pixeles limpio y con ruido ~1e-7 en la salida del encoder
# (dos semillas de ruido). Si la loss ruidosa se separa de la limpia en el
# mismo orden (~1e-4) que la corrida con embeddings precalculados (job
# 30552), esa diferencia es amplificacion de redondeo, no un efecto real.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env
cd $HOME/MaskDP_mmodal
export NVIDIA_TF32_OVERRIDE=0
OUT=$HOME/models/e2e_noise/$SLURM_JOB_ID

for RUN in clean:0:0 noise1:1e-7:1 noise2:1e-7:2; do
    IFS=: read NAME STD NSEED <<< "$RUN"
    E2E_NOISE_STD=$STD E2E_NOISE_SEED=$NSEED HYDRA_FULL_ERROR=1 python e2e_noise_pretrain_coinrun.py \
        agent=bc_uni_coinrun task=coinrun seed=1 \
        replay_buffer_dir=$HOME/MaskDP_mmodal/data \
        pad_short_episodes=true num_grad_steps=301 log_every_steps=10 \
        "snapshots=[]" snapshot_dir=$OUT/snap_$NAME \
        hydra.run.dir=$OUT/$NAME \
        agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
        use_wandb=false use_tb=true exp_name=e2e_$NAME
done

python - "$OUT" <<'PY'
import sys, csv, numpy as np
out = sys.argv[1]
read = lambda d: {int(float(r["step"])): float(r["action_loss"]) for r in csv.DictReader(open(f"{out}/{d}/train.csv"))}
clean, n1, n2 = read("clean"), read("noise1"), read("noise2")
steps = sorted(clean)
for s in steps[::5] + [steps[-1]]:
    print(f"paso {s:4d}: limpio {clean[s]:.6f}  ruido1 {n1[s]-clean[s]:+.2e}  ruido2 {n2[s]-clean[s]:+.2e}")
for name, n in [("ruido1", n1), ("ruido2", n2)]:
    d = np.array([n[s] - clean[s] for s in steps])
    print(f"{name}: max |diff| {np.abs(d).max():.2e}, diff>0 en {(d > 0).sum()}/{len(d)} puntos")
PY
