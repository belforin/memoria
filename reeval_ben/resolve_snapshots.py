"""
Lista los snapshots DMC finales de Benjamin (sus listas
slurms/eval/final/{uni_all,hier_all}.txt) para la re-evaluacion con la
alineacion corregida (belforin/memoria METODOLOGIA_DT_HDT.md seccion 7.8).
Resuelve cada ruta con la misma logica que sus hier_return.sh /
maskdp_return.sh (si la ruta no nombra el dominio, busca
<ruta>/{cheetah,walker,quadruped}/<seed>) y escribe un TSV:
arch  cond  domain  task  seed  repo  snapshot
"""
import argparse
import re
from pathlib import Path

FINAL = Path("/home/bmancilla/archive/MaskDP/slurms/eval/final")
ARCHIVE = Path("/home/bmancilla/archive/MaskDP")
TASKS = {"cheetah": "cheetah_run", "walker": "walker_run", "quadruped": "quadruped_walk"}
CONDS = {"notrick": "sin_aumentacion", "j_md": "ambas"}
STEP = 400000


def resolve(snap_dir, seed):
    p = Path(snap_dir)
    if p.is_file():
        p = p.parent
    if not re.search(r"cheetah|walker|quadruped", str(p)):
        for dom in TASKS:
            if (p / dom / str(seed)).is_dir():
                p = p / dom / str(seed)
                break
    m = re.search(r"(cheetah|walker|quadruped)", str(p))
    return p, (m.group(1) if m else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="snapshots.tsv")
    args = ap.parse_args()
    rows, missing = [], []
    for list_name, repo in [("uni_all.txt", "MaskDP_paper"), ("hier_all.txt", "Hierarchical")]:
        for line in (FINAL / list_name).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, seed, snap_dir = line.split()[:3]
            arch, cond = None, None
            for c in CONDS:
                if name.endswith("_" + c):
                    arch, cond = name[: -len(c) - 1], CONDS[c]
            if cond is None:
                continue  # solo jitter / solo mdrop: fuera del alcance
            p, dom = resolve(snap_dir, seed)
            snap = p / f"snapshot_{STEP}.pt"
            if dom is None or not snap.is_file():
                missing.append(f"{name} s{seed}: {snap}")
                continue
            arch = "unistream" if arch == "uni" else arch
            rows.append((arch, cond, dom, TASKS[dom], seed, str(ARCHIVE / repo), str(snap)))
    rows = sorted(set(rows))
    with open(args.out, "w") as f:
        f.write("arch\tcond\tdomain\ttask\tseed\trepo\tsnapshot\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(rows)} snapshots -> {args.out}")
    for m in missing:
        print("[falta]", m)


if __name__ == "__main__":
    main()
