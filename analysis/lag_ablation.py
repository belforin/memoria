"""
Perfil de dependencia por intervencion, por retraso y modalidad, para BC-uni/
BC-hier (belforin/memoria METODOLOGIA_DT_HDT.md seccion 7.16, metrica 2).

Para cada ventana del dataset (K = traj_length del snapshot, la misma que en
evaluacion) se mira la prediccion de la accion actual a_t (ultima posicion).
Para cada retraso k y modalidad se reemplaza UN token de entrada por el de otro
paso del dataset elegido al azar (resample ablation: valores realistas, no
ceros ni blur) y se mide cuanto cambia la prediccion:
  - estados s_{t-k}, k = 0..K-1 (k = 0 es el estado actual);
  - acciones a_{t-k}, k = 1..K-1 (a_t no es entrada).
Ademas: todas las acciones pasadas a la vez y todos los estados pasados a la
vez (dependencia de la historia completa de cada modalidad).

Distancia:
  - CoinRun (acciones discretas): KL(p || p') entre las distribuciones
    softmax (temperatura 1, la de evaluacion) y tasa de cambio del argmax;
  - DMC (continuas): RMS de la diferencia de la accion predicha (tanh).

La intervencion se hace en la entrada del modelo, asi que se define igual
para las dos topologias (no depende de capas). En CoinRun se reemplaza el
embedding del IMPALA congelado (funcion fija de cada frame, seccion 7.4), que
equivale a reemplazar el frame.

Tambien guarda la metrica 3b (copycat, Wen et al. 2020) sobre las mismas
ventanas: cuanto se parece la prediccion a la accion anterior a_{t-1}, frente
a cuanto se parece la accion del experto a_t a a_{t-1}.

Uso:
  python analysis/lag_ablation.py --snapshot <snapshot.pt> --data_dir <dir de npz> \
      --out <prefijo de salida> [--n_windows 512 --n_donors 4 --seed 0]
Escribe <prefijo>.npz (por ventana) y <prefijo>.json (resumen).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.bc_ar import BCARModel  # noqa: E402


def load_model(path, device):
    payload = torch.load(path, map_location=device)
    cfg = payload["cfg"]
    pixel = bool(cfg.get("use_pixel_obs", False))
    discrete = bool(cfg.get("discrete_actions", False))
    sd = payload["model"]
    obs_dim = None if pixel else int(sd["state_embed.weight"].shape[1])
    action_dim = int(cfg.num_actions) if discrete else int(sd["action_embed.weight"].shape[1])
    model = BCARModel(obs_dim, action_dim, cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


def episode_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    assert files, f"sin episodios en {data_dir}"
    return files


def sample_windows(files, K, n_windows, n_pool, rng):
    """
    Ventanas con la alineacion del replay buffer de Benjamin
    (has_dummy_transition=True): obs[i : i+K] con action[i+1 : i+K+1];
    a_t de la ultima posicion es action[i+K]. Solo ventanas completas dentro
    del episodio (sin relleno). Tambien devuelve un pool de pasos sueltos
    (obs, accion) de OTROS episodios, para usar como donantes.
    """
    order = rng.permutation(len(files))
    obs_w, act_w, tgt_w, prev_w = [], [], [], []
    used = 0
    for fi in order:
        if len(obs_w) >= n_windows:
            break
        ep = np.load(files[fi])
        L = ep["observation"].shape[0] - 1
        used += 1
        if L < K + 1:
            continue
        i = int(rng.integers(0, L - K + 1))
        obs_w.append(ep["observation"][i:i + K])
        act_w.append(ep["action"][i + 1:i + K])          # a_{t-K+1} .. a_{t-1}
        tgt_w.append(ep["action"][i + K])                # a_t del experto
        prev_w.append(ep["action"][i + K - 1])           # a_{t-1}
    assert len(obs_w) == n_windows, f"solo {len(obs_w)} ventanas de largo {K}"

    pool_obs, pool_act = [], []
    for fi in order[used:]:
        if len(pool_obs) >= n_pool:
            break
        ep = np.load(files[fi])
        L = ep["observation"].shape[0] - 1
        for j in rng.choice(np.arange(1, L + 1), size=min(8, L), replace=False):
            pool_obs.append(ep["observation"][j - 1])
            pool_act.append(ep["action"][j])
    assert len(pool_obs) >= n_pool, f"pool de donantes chico: {len(pool_obs)}"
    return (np.stack(obs_w), np.stack(act_w), np.stack(tgt_w), np.stack(prev_w),
            np.stack(pool_obs[:n_pool]), np.stack(pool_act[:n_pool]))


@torch.no_grad()
def to_inputs(model, obs, act, device, chunk=64):
    """obs del dataset -> entrada del modelo (en CoinRun, embeddings del IMPALA congelado)"""
    obs_t = torch.as_tensor(obs, device=device)
    if model.pixel_obs:
        shp = obs_t.shape
        flat = obs_t.reshape(-1, *shp[-3:])
        emb = torch.cat([model.embed_obs(flat[c:c + chunk].unsqueeze(1))[:, 0]
                         for c in range(0, flat.shape[0], chunk)])
        obs_t = emb.reshape(*shp[:-3], -1)
    else:
        obs_t = obs_t.float()
    act_t = torch.as_tensor(act, device=device)
    act_t = act_t.long() if model.discrete_actions else act_t.float()
    return obs_t, act_t


@torch.no_grad()
def predict_last(model, obs, act, chunk=256):
    """obs (B, K, ...), act (B, K-1, ...) -> prediccion en la ultima posicion (B, out)"""
    return torch.cat([model(obs[c:c + chunk], act[c:c + chunk])[:, -1] for c in range(0, obs.shape[0], chunk)])


def distance(model, base, pert):
    """(B, out) x (B, out) -> (dist (B,), flip (B,) o None)"""
    if model.discrete_actions:
        lp, lq = F.log_softmax(base, -1), F.log_softmax(pert, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        flip = (base.argmax(-1) != pert.argmax(-1)).float()
        return kl, flip
    return ((base - pert) ** 2).mean(-1).sqrt(), None


@torch.no_grad()
def lag_profile(model, obs, act, pool_obs, pool_act, n_donors, rng):
    """
    obs (N, K, ...), act (N, K-1, ...) ya en formato de entrada del modelo.
    Devuelve dict con d_state (K, N), d_act (K, N) (fila 0 = NaN), los mismos
    para el flip (CoinRun), y d_all_past_states / d_all_past_actions (N,).
    Cada valor es el promedio sobre n_donors reemplazos.
    """
    N, K = obs.shape[0], obs.shape[1]
    base = predict_last(model, obs, act)
    base_r = base.repeat(n_donors, 1)
    obs_r = obs.repeat(n_donors, *[1] * (obs.dim() - 1))
    act_r = act.repeat(n_donors, *[1] * (act.dim() - 1))
    P = pool_obs.shape[0]

    def run(o, a):
        d, f = distance(model, base_r, predict_last(model, o, a))
        d = d.reshape(n_donors, N).mean(0)
        f = None if f is None else f.reshape(n_donors, N).mean(0)
        return d, f

    def donors(n):
        return torch.as_tensor(rng.integers(0, P, size=n), device=obs.device)

    out = {k: np.full((K, N), np.nan, np.float32) for k in ["d_state", "d_act", "f_state", "f_act"]}
    for k in range(K):
        pos = K - 1 - k                       # s_{t-k} en la posicion K-1-k
        o = obs_r.clone()
        o[:, pos] = pool_obs[donors(o.shape[0])]
        d, f = run(o, act_r)
        out["d_state"][k] = d.cpu().numpy()
        if f is not None:
            out["f_state"][k] = f.cpu().numpy()
        if k >= 1:                            # a_{t-k} es act[:, K-1-k]
            a = act_r.clone()
            a[:, K - 1 - k] = pool_act[donors(a.shape[0])]
            d, f = run(obs_r, a)
            out["d_act"][k] = d.cpu().numpy()
            if f is not None:
                out["f_act"][k] = f.cpu().numpy()

    o = obs_r.clone()
    idx = donors(o.shape[0] * (K - 1)).reshape(o.shape[0], K - 1)
    o[:, :K - 1] = pool_obs[idx]
    out["d_all_past_states"] = run(o, act_r)[0].cpu().numpy()
    a = act_r.clone()
    idx = donors(a.shape[0] * (K - 1)).reshape(a.shape[0], K - 1)
    a[:] = pool_act[idx]
    out["d_all_past_actions"] = run(obs_r, a)[0].cpu().numpy()
    out["base"] = base.cpu().numpy()
    return out


def copycat(model, base, tgt, prev):
    """metrica 3b: prediccion vs a_{t-1}, frente a experto a_t vs a_{t-1}"""
    if model.discrete_actions:
        pred = base.argmax(-1)
        tgt, prev = tgt.reshape(-1), prev.reshape(-1)
        return dict(pred_eq_prev=float((pred == prev).mean()), expert_eq_prev=float((tgt == prev).mean()),
                    pred_eq_expert=float((pred == tgt).mean()))
    rms = lambda x, y: float(np.sqrt(((x - y) ** 2).mean(-1)).mean())
    return dict(rms_pred_prev=rms(base, prev), rms_expert_prev=rms(tgt, prev), rms_pred_expert=rms(base, tgt))


def summarize(prof):
    """escalares por modelo, a partir de las medias por retraso"""
    ms = np.nanmean(prof["d_state"], 1)
    ma = np.nanmean(prof["d_act"], 1)
    ma0 = np.nan_to_num(ma)
    K = ms.shape[0]
    lags = np.arange(K)
    tot = ms + ma0
    cum = np.cumsum(tot) / tot.sum()
    past_s, past_a = ms[1:].sum(), ma0[1:].sum()
    return dict(
        K=K,
        state_curve=ms.tolist(), action_curve=ma.tolist(),
        d_current_state=float(ms[0]),
        d_all_past_states=float(np.mean(prof["d_all_past_states"])),
        d_all_past_actions=float(np.mean(prof["d_all_past_actions"])),
        # fraccion de la suma de efectos individuales; los efectos no son
        # aditivos, es un reparto aproximado
        frac_action_past=float(past_a / (past_s + past_a)),
        frac_action_all=float(ma0.sum() / tot.sum()),
        mean_lag=float((lags * tot).sum() / tot.sum()),
        mean_lag_past=float((lags[1:] * tot[1:]).sum() / tot[1:].sum()),
        lag_90=int(np.searchsorted(cum, 0.9)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=512)
    ap.add_argument("--n_donors", type=int, default=4)
    ap.add_argument("--n_pool", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cfg = load_model(args.snapshot, args.device)
    K = int(cfg.traj_length)
    # misma semilla de ventanas para todos los modelos: mismas entradas
    rng = np.random.default_rng(args.seed)
    obs, act, tgt, prev, p_obs, p_act = sample_windows(
        episode_files(args.data_dir), K, args.n_windows, args.n_pool, rng)
    obs_t, act_t = to_inputs(model, obs, act, args.device)
    p_obs_t, p_act_t = to_inputs(model, p_obs[:, None], p_act[:, None], args.device)
    prof = lag_profile(model, obs_t, act_t, p_obs_t[:, 0], p_act_t[:, 0], args.n_donors,
                       np.random.default_rng(args.seed + 1))

    summary = summarize(prof)
    summary.update(copycat(model, prof["base"], tgt, prev))
    if model.discrete_actions:
        summary["flip_state_curve"] = np.nanmean(prof["f_state"], 1).tolist()
        summary["flip_action_curve"] = np.nanmean(prof["f_act"], 1).tolist()
    summary.update(snapshot=args.snapshot, data_dir=args.data_dir, topology=model.topology,
                   n_windows=args.n_windows, n_donors=args.n_donors, seed=args.seed,
                   metric="KL" if model.discrete_actions else "RMS")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out + ".npz", **{k: v for k, v in prof.items() if k != "base"})
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    keys = ["d_current_state", "d_all_past_states", "d_all_past_actions", "frac_action_past",
            "mean_lag_past", "lag_90"]
    print(" | ".join(f"{k}={summary[k]:.4g}" for k in keys))


if __name__ == "__main__":
    main()
