"""
Corre el eval_return.py de Benjamin, dentro de su repo y sin modificarlo, con
su agente de evaluacion original o con la alineacion corregida
(belforin/memoria METODOLOGIA_DT_HDT.md seccion 7.8).

Uso: python eval_return_aligned.py --repo <MaskDP_paper|Hierarchical> --mode original|fixed -- <overrides de hydra>

--mode fixed cambia solo dos cosas de MaskingEvalAgent / MaskingEvalAgentMultimodal:
- la memoria de acciones guarda T_cond-1 acciones (una por detras de la de
  estados), asi la accion en la posicion 2k+1 es la tomada en el estado de la
  posicion 2k;
- in_warmup siempre es True: en cada paso se lee la prediccion de la posicion
  del estado actual (pred_a[n_estados-1]), que es lo que hace su agente
  durante el calentamiento y lo que describe su Algoritmo 1.
Con replan_freq=1 (su evaluacion de lazo cerrado) no hay cola de acciones.
"""
import argparse
import os
import runpy
import sys
from collections import deque

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--mode", choices=["original", "fixed"], required=True)
args, overrides = ap.parse_known_args()
overrides = [o for o in overrides if o != "--"]

os.chdir(args.repo)
sys.path.insert(0, args.repo)

import agent.mdp_return as mdp_return  # noqa: E402  (modulo de su repo)

cls = getattr(mdp_return, "MaskingEvalAgentMultimodal", None) or mdp_return.MaskingEvalAgent

if args.mode == "fixed":
    _reset = cls.reset

    def reset(self):
        _reset(self)
        assert self.T_cond >= 2
        self._action_buffer = deque(maxlen=self.T_cond - 1)

    cls.reset = reset
    cls.in_warmup = property(lambda self: True)

print(f"[eval_return_aligned] repo={args.repo} clase={cls.__name__} mode={args.mode}", flush=True)
sys.argv = [os.path.join(args.repo, "eval_return.py")] + overrides
runpy.run_path(sys.argv[0], run_name="__main__")
