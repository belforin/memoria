"""
Metrica 4 de interpretabilidad (belforin/memoria METODOLOGIA_DT_HDT.md seccion
7.16): atribucion de todo el modelo sobre los tokens de entrada con
Integrated Gradients (Sundararajan et al. 2017), y su fidelidad con curvas de
eliminacion (AOPC, DeYoung et al. 2020) contra otros ordenes de tokens.

Mismas ventanas y donantes que analysis/lag_ablation.py (--seed 0).

IG: sobre los embeddings de entrada (antes de la posicion) de estados y
acciones, BCARModel.embed_tokens/forward_tokens. Linea base = token medio de
cada modalidad en el pool de donantes (no ceros: el embedding cero queda fuera
de distribucion). Suma de Riemann en el punto medio con m pasos (default 128:
la completitud converge del orden de 1/m, se reporta en el resumen). Objetivo:
logit de la accion argmax en CoinRun (como AttAttr/SARFA); en DMC, una IG por
dimension de accion y |.| sumado sobre dimensiones. Atribucion de un token =
suma sobre las dimensiones del embedding; se reporta |.|.

AOPC: para cada orden de tokens (de mas a menos importante) se reemplazan los
top-k tokens por donantes de otros episodios (k en --ks) y se mide la misma
distancia que en lag_ablation (KL en CoinRun, RMS en DMC); AOPC = media sobre
k. Mas alto = el orden encuentra antes los tokens que importan. Ordenes:
  ig        |IG| por token;
  occlusion efecto de reemplazar ese token solo (metrica 2), por ventana;
  recency   s_t, a_{t-1}, s_{t-1}, a_{t-2}, ... (linea base: "lo reciente
            importa mas");
  random    permutacion al azar.
Tambien AOPC solo sobre la historia (s_t fijo): en DMC s_t queda primero en
todos los ordenes salvo random y satura la distancia.

Uso:
  python analysis/input_attribution.py --snapshot <snap.pt> --data_dir <dir> --out <prefijo>
Escribe <prefijo>.npz y <prefijo>.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.lag_ablation import (distance, episode_files, lag_profile, load_model,  # noqa: E402
                                   predict_last, sample_windows, to_inputs)


def integrated_gradients(model, s, a, s_base, a_base, m=32, chunk=64):
    """
    s, a: (N, K, D) tokens de entrada (a con el relleno de a_t en la posicion K-1).
    s_base, a_base: (D,) linea base. Devuelve (attr_s (N, K), attr_a (N, K-1),
    completitud (N,) = |sum attr - (F(x) - F(base))| / |F(x) - F(base)| promediada
    sobre objetivos).
    """
    N, K, D = s.shape
    alphas = (torch.arange(m, device=s.device, dtype=s.dtype) + 0.5) / m
    with torch.no_grad():
        pred = model.forward_tokens(s, a)[:, -1]
    if model.discrete_actions:
        targets = [pred.argmax(-1)]                     # un objetivo por ventana
    else:
        targets = list(range(pred.shape[-1]))           # una IG por dimension
    attr_s = torch.zeros(N, K, device=s.device)
    attr_a = torch.zeros(N, K - 1, device=s.device)
    comp = torch.zeros(N, device=s.device)

    sb = s_base.expand_as(s)
    ab = a_base.expand_as(a).clone()
    ab[:, K - 1] = a[:, K - 1]                          # relleno de a_t: no es entrada

    def target_value(out, tgt, idx):
        if model.discrete_actions:
            return out.gather(-1, tgt[idx].repeat_interleave(m)[:, None])[:, 0]
        return out[:, tgt]

    for tgt in targets:
        for c in range(0, N, chunk):
            idx = slice(c, min(c + chunk, N))
            n = idx.stop - idx.start
            ds, da = s[idx] - sb[idx], a[idx] - ab[idx]
            si = (sb[idx][:, None] + alphas[None, :, None, None] * ds[:, None]).reshape(n * m, K, D)
            ai = (ab[idx][:, None] + alphas[None, :, None, None] * da[:, None]).reshape(n * m, K, D)
            si.requires_grad_(True)
            ai.requires_grad_(True)
            out = model.forward_tokens(si, ai)[:, -1]
            gs, ga = torch.autograd.grad(target_value(out, tgt, idx).sum(), [si, ai])
            ig_s = (gs.reshape(n, m, K, D).mean(1) * ds).sum(-1)
            ig_a = (ga.reshape(n, m, K, D).mean(1) * da).sum(-1)[:, :K - 1]
            with torch.no_grad():
                fx = model.forward_tokens(s[idx], a[idx])[:, -1]
                fb = model.forward_tokens(sb[idx], ab[idx])[:, -1]
                if model.discrete_actions:
                    t = tgt[idx][:, None]
                    delta = (fx.gather(-1, t) - fb.gather(-1, t))[:, 0]
                else:
                    delta = fx[:, tgt] - fb[:, tgt]
                total = ig_s.sum(-1) + ig_a.sum(-1)
                comp[idx] += (total - delta).abs() / delta.abs().clamp(min=1e-6)
            attr_s[idx] += ig_s.detach().abs()
            attr_a[idx] += ig_a.detach().abs()
    return attr_s, attr_a, comp / len(targets)


def by_lag(attr_s, attr_a):
    """(N, K), (N, K-1) por posicion -> (N, K) por retraso k (s) y (N, K) (a, k=0 NaN)"""
    N, K = attr_s.shape
    s_lag = attr_s.flip(1)
    a_lag = torch.full((N, K), float("nan"), device=attr_s.device)
    a_lag[:, 1:] = attr_a.flip(1)
    return s_lag, a_lag


@torch.no_grad()
def aopc(model, obs, act, pool_obs, pool_act, scores, ks, n_donors, rng):
    """
    scores: (N, 2K-1) importancia por token, tokens ordenados como
    [s_0..s_{K-1} (posiciones), a_0..a_{K-2}]. Reemplaza los top-k por
    donantes. Devuelve (len(ks),) distancias medias.
    """
    N, K = obs.shape[0], obs.shape[1]
    base = predict_last(model, obs, act).repeat(n_donors, 1)
    order = torch.argsort(scores, dim=1, descending=True).repeat(n_donors, 1)
    obs_r = obs.repeat(n_donors, *[1] * (obs.dim() - 1))
    act_r = act.repeat(n_donors, *[1] * (act.dim() - 1))
    P = pool_obs.shape[0]
    rows = torch.arange(obs_r.shape[0], device=obs.device)
    out = []
    for k in ks:
        o, a = obs_r.clone(), act_r.clone()
        for r in range(k):
            tok = order[:, r]
            is_s = tok < K
            d_idx = torch.as_tensor(rng.integers(0, P, size=o.shape[0]), device=obs.device)
            o[rows[is_s], tok[is_s]] = pool_obs[d_idx[is_s]]
            a[rows[~is_s], tok[~is_s] - K] = pool_act[d_idx[~is_s]]
        d, _ = distance(model, base, predict_last(model, o, a))
        out.append(float(d.mean()))
    return np.array(out)


def history_only(scores):
    """scores (N, 2K-1) con s_t (posicion K-1) al final del orden: el AOPC no lo reemplaza mientras k < 2K-1"""
    K = (scores.shape[1] + 1) // 2
    sc = scores.clone()
    sc[:, K - 1] = -float("inf")
    return sc


def lag_scalars(s_lag, a_lag):
    """mismas definiciones que lag_ablation.summarize, sobre curvas medias"""
    ms = s_lag.mean(0).cpu().numpy()
    ma = np.nan_to_num(np.nanmean(a_lag.cpu().numpy(), 0))
    lags = np.arange(ms.shape[0])
    tot = ms + ma
    cum = np.cumsum(tot) / tot.sum()
    return dict(state_curve=ms.tolist(), action_curve=ma.tolist(),
                frac_action_past=float(ma[1:].sum() / (ms[1:].sum() + ma[1:].sum())),
                mean_lag_past=float((lags[1:] * tot[1:]).sum() / tot[1:].sum()),
                lag_90=int(np.searchsorted(cum, 0.9)))


def spearman_rows(x, y):
    """Spearman por fila entre (N, T) y (N, T)"""
    rx = torch.argsort(torch.argsort(x, 1), 1).float()
    ry = torch.argsort(torch.argsort(y, 1), 1).float()
    rx, ry = rx - rx.mean(1, keepdim=True), ry - ry.mean(1, keepdim=True)
    return (rx * ry).sum(1) / (rx.norm(dim=1) * ry.norm(dim=1)).clamp(min=1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=512)
    ap.add_argument("--n_donors", type=int, default=4)
    ap.add_argument("--n_pool", type=int, default=4096)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cfg = load_model(args.snapshot, args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    K = int(cfg.traj_length)
    rng = np.random.default_rng(args.seed)
    obs, act, _, _, p_obs, p_act = sample_windows(episode_files(args.data_dir), K, args.n_windows, args.n_pool, rng)
    obs_t, act_t = to_inputs(model, obs, act, args.device)
    p_obs_t, p_act_t = to_inputs(model, p_obs[:, None], p_act[:, None], args.device)

    with torch.no_grad():
        s, a = model.embed_tokens(obs_t, act_t)
        ps, pa = model.embed_tokens(p_obs_t, p_act_t)
    # ~8192 secuencias-paso por lote (m * K tokens por ventana): entra en 24 GB
    chunk = max(1, 8192 // (args.m * K))
    attr_s, attr_a, comp = integrated_gradients(model, s, a, ps.mean((0, 1)), pa.mean((0, 1)), m=args.m, chunk=chunk)
    s_lag, a_lag = by_lag(attr_s, attr_a)

    # oclusion de un token: la metrica 2 sobre las mismas ventanas
    prof = lag_profile(model, obs_t, act_t, p_obs_t[:, 0], p_act_t[:, 0], args.n_donors,
                       np.random.default_rng(args.seed + 1))
    occ_s = torch.as_tensor(prof["d_state"][::-1].T.copy(), device=args.device)       # (N, K) por posicion
    occ_a = torch.as_tensor(prof["d_act"][1:][::-1].T.copy(), device=args.device)     # (N, K-1)

    N = obs_t.shape[0]
    ig_tok = torch.cat([attr_s, attr_a], 1)
    occ_tok = torch.cat([occ_s, occ_a], 1)
    # recencia: s_t, a_{t-1}, s_{t-1}, ...; puntaje = -(orden)
    rec = torch.empty(2 * K - 1, device=args.device)
    rec[:K] = -(2 * torch.arange(K - 1, -1, -1, device=args.device)).float()          # s en pos p: retraso K-1-p
    rec[K:] = -(2 * torch.arange(K - 1, 0, -1, device=args.device) - 1).float()       # a en pos p: retraso K-1-p
    rec_tok = rec.expand(N, -1)
    rand_tok = torch.as_tensor(np.random.default_rng(args.seed + 2).random((N, 2 * K - 1)),
                               device=args.device, dtype=torch.float32)

    ks = [int(k) for k in args.ks.split(",") if int(k) <= 2 * K - 1]
    curves, curves_hist = {}, {}
    for name, sc in [("ig", ig_tok), ("occlusion", occ_tok), ("recency", rec_tok), ("random", rand_tok)]:
        curves[name] = aopc(model, obs_t, act_t, p_obs_t[:, 0], p_act_t[:, 0], sc, ks, args.n_donors,
                            np.random.default_rng(args.seed + 3))
        # solo historia: s_t nunca se reemplaza. En DMC todos los ordenes (salvo random) ponen s_t
        # primero y su efecto satura la distancia, asi que el AOPC completo no los distingue
        curves_hist[name] = aopc(model, obs_t, act_t, p_obs_t[:, 0], p_act_t[:, 0], history_only(sc), ks,
                                 args.n_donors, np.random.default_rng(args.seed + 3))

    summary = dict(ig=lag_scalars(s_lag, a_lag),
                   ig_completeness_rel_err=float(comp.median()),
                   spearman_ig_occlusion=float(spearman_rows(ig_tok, occ_tok).mean()),
                   aopc_ks=ks, aopc_curves={k: v.tolist() for k, v in curves.items()},
                   aopc={k: float(v.mean()) for k, v in curves.items()},
                   aopc_hist_curves={k: v.tolist() for k, v in curves_hist.items()},
                   aopc_hist={k: float(v.mean()) for k, v in curves_hist.items()},
                   snapshot=args.snapshot, data_dir=args.data_dir, topology=model.topology,
                   n_windows=args.n_windows, m=args.m, seed=args.seed,
                   metric="KL" if model.discrete_actions else "RMS")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out + ".npz", ig_state_lag=s_lag.cpu().numpy(), ig_action_lag=a_lag.cpu().numpy(),
                        completeness=comp.cpu().numpy())
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"IG frac_action_past={summary['ig']['frac_action_past']:.3f} mean_lag_past="
          f"{summary['ig']['mean_lag_past']:.2f} | completitud (err rel. mediano)={summary['ig_completeness_rel_err']:.3g}"
          f" | Spearman(IG, oclusion)={summary['spearman_ig_occlusion']:.3f} | AOPC "
          + " ".join(f"{k}={v:.4g}" for k, v in summary["aopc"].items())
          + " | AOPC historia " + " ".join(f"{k}={v:.4g}" for k, v in summary["aopc_hist"].items()))


if __name__ == "__main__":
    main()
