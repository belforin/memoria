#!/bin/bash
#SBATCH --job-name=maskdp_e2e_emb_vs_pix
#SBATCH --partition=ialab-high
#SBATCH --qos=regular
#SBATCH --account=default-account
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment="Verificacion de que entrenar behaviour cloning sobre embeddings precalculados equivale a entrenar sobre pixeles en CoinRun"

# Etapa BC (METODOLOGIA_DT_HDT.md seccion 7.4): 300 pasos de BC-uni con la
# misma semilla sobre pixeles (data/) y sobre embeddings (data_emb/); mismos
# archivos en el mismo orden => mismos batches. Compara action_loss cada 10
# pasos. TF32 desactivado en todo el proceso para comparar FP32 vs FP32.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maskdp-env
cd $HOME/MaskDP_mmodal
export NVIDIA_TF32_OVERRIDE=0
OUT=$HOME/models/e2e_emb_vs_pix/$SLURM_JOB_ID

for DATA in data data_emb; do
    HYDRA_FULL_ERROR=1 python pretrain_coinrun.py \
        agent=bc_uni_coinrun task=coinrun seed=1 \
        replay_buffer_dir=$HOME/MaskDP_mmodal/$DATA \
        pad_short_episodes=true num_grad_steps=301 log_every_steps=10 \
        "snapshots=[]" snapshot_dir=$OUT/snap_$DATA \
        hydra.run.dir=$OUT/$DATA \
        agent.transformer_cfg.pretrained_encoder_path=$HOME/MaskDP_mmodal/pretrained_encoders/procgen_coinrun_easy_encoder.pt \
        use_wandb=false use_tb=true exp_name=e2e_$DATA
done

python - "$OUT" <<'PY'
import sys, csv, numpy as np
out = sys.argv[1]
read = lambda d: {int(float(r["step"])): float(r["action_loss"]) for r in csv.DictReader(open(f"{out}/{d}/train.csv"))}
pix, emb = read("data"), read("data_emb")
steps = sorted(set(pix) & set(emb))
diff = np.array([abs(pix[s] - emb[s]) for s in steps])
for s in steps[::5] + [steps[-1]]:
    print(f"paso {s:4d}: pixeles {pix[s]:.6f}  embeddings {emb[s]:.6f}  |diff| {abs(pix[s]-emb[s]):.2e}")
print(f"max |diff| sobre {len(steps)} puntos: {diff.max():.2e}; loss pixeles {pix[steps[0]]:.4f} -> {pix[steps[-1]]:.4f}")
PY
