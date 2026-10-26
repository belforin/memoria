#!/bin/bash
# Métricas SARFA + AttAttr (§7.11) sobre bc_uni, bc_hier, bc_hier223, bc_hier223_enc128.
set -e
export PYTHONPATH=$HOME/MaskDP_bc
cd $HOME/MaskDP_bc
R=/home/belforin/models/bc_plugandplay/interp_procgen; O=/home/belforin/models/bc_plugandplay/interp_procgen/metrics_4modelos; mkdir -p $O
[ -z "$SKIP_ATTATTR" ] && python analysis/bc_attn_attr_metrics.py --out-dir $O $(ls -d $R/bc_*_s[123]/attattr)
$HOME/miniconda3/envs/maskdp-env/bin/python analysis/sarfa_entropy_summary.py $R/sarfa --out-csv $O/sarfa_entropy_answer_maps.csv
$HOME/miniconda3/envs/maskdp-env/bin/python analysis/sarfa_topk_concentration.py $R/sarfa --out-dir $O
$HOME/miniconda3/envs/maskdp-env/bin/python analysis/sarfa_temporal_recency.py $R/sarfa --out-dir $O
