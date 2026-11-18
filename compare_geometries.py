"""Comparación de geometrías de adquisición en 2D: PINN y tomografía convencional.

Las 4 geometrías tienen el mismo número de datos (12 fuentes x 52 receptores = 624 tiempos)
y el mismo ruido (1 ms). Para cada una:
  * se generan los datos con FMM sobre el modelo verdadero (guardados en data/geometrias/<nombre>/),
  * se invierte con la PINN (misma configuración que run_tomography.py),
  * se invierte con rayos curvos (lam elegido con el principio de discrepancia),
  * se trazan los rayos en el modelo verdadero para ver la cobertura.

    python compare_geometries.py --geom superficie     # una geometría (~3-5 min)
    python compare_geometries.py --summary              # tabla y figura con todas
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from pinn_tomo.conventional import invert, trace_rays
from pinn_tomo.forward import (GEOMETRIES, Grid, anomaly_metrics, eikonal_fmm, geometry,
                               traveltimes, true_velocity)
from pinn_tomo.pinn import TauNet, VelocityNet, eikonal_residual, traveltime

OUT = Path("outputs/geometrias")
DATA = Path("data/geometrias")
NOISE = 1e-3
LAMS = [0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]


def make_data(name, grid, v_true):
    d = DATA / name
    src, rec = geometry(name)
    if (d / "tiempos.csv").exists():
        tab = np.loadtxt(d / "tiempos.csv", delimiter=",", skiprows=1)
        t_obs = np.zeros((len(src), len(rec)))
        t_obs[tab[:, 0].astype(int), tab[:, 3].astype(int)] = tab[:, 6]
        return src, rec, t_obs
    rng = np.random.default_rng(0)
    t_clean, _ = traveltimes(v_true, grid, src, rec)
    t_obs = t_clean + NOISE * rng.standard_normal(t_clean.shape)
    d.mkdir(parents=True, exist_ok=True)
    fmt = ["%d", "%.4f", "%.4f"]
    np.savetxt(d / "fuentes.csv", np.c_[np.arange(len(src)), src], delimiter=",", header="id,x_km,z_km", comments="", fmt=fmt)
    np.savetxt(d / "receptores.csv", np.c_[np.arange(len(rec)), rec], delimiter=",", header="id,x_km,z_km", comments="", fmt=fmt)
    rows = [(i, *src[i], j, *rec[j], t_obs[i, j], t_clean[i, j]) for i in range(len(src)) for j in range(len(rec))]
    np.savetxt(d / "tiempos.csv", np.array(rows), delimiter=",",
               header="id_fuente,xs_km,zs_km,id_receptor,xr_km,zr_km,t_obs_s,t_sin_ruido_s", comments="",
               fmt=fmt + fmt + ["%.6f", "%.6f"])
    (d / "info.json").write_text(json.dumps({"geometria": GEOMETRIES[name], "ruido_desv_std_s": NOISE, "semilla": 0,
                                             "n_tiempos": int(t_obs.size)}, indent=2, ensure_ascii=False))
    return src, rec, t_obs


def train_pinn(src, rec, t_obs, iters=8000, lr=2e-3, n_colloc=3000, w_pde=100.0, vmin=1.0, vmax=4.0, seed=0):
    """Misma configuración por defecto que run_tomography.py."""
    torch.manual_seed(seed)
    S = torch.tensor(np.repeat(src, len(rec), axis=0), dtype=torch.float32)
    R = torch.tensor(np.tile(rec, (len(src), 1)), dtype=torch.float32)
    T_obs = torch.tensor(t_obs.reshape(-1, 1), dtype=torch.float32)
    src_t = torch.tensor(src, dtype=torch.float32)
    t_scale = float(T_obs.abs().mean())
    v_net = VelocityNet(vmin, vmax)
    tau_net = TauNet(s_ref=2.0 / (vmin + vmax))
    opt = torch.optim.Adam(list(v_net.parameters()) + list(tau_net.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.02)
    for it in range(iters + 1):
        xz = torch.rand(n_colloc, 2)
        sxz = src_t[torch.randint(len(src_t), (n_colloc,))]
        l_pde = (eikonal_residual(tau_net, v_net, xz, sxz) ** 2).mean()
        l_data = (((traveltime(tau_net, R, S) - T_obs) / t_scale) ** 2).mean() * 1e3
        loss = w_pde * l_pde + l_data
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 2000 == 0:
            print(f"  PINN it {it:5d} | pde {l_pde.item():.2e} | data {l_data.item():.2e}", flush=True)
    return v_net


def fmm_times(v, grid, src, rec):
    return traveltimes(v, grid, src, rec)[0]


def summarize(v, t_calc, t_obs, v_true, grid, seconds):
    return {"rms_tiempos_ms": float(1e3 * np.sqrt(np.mean((t_calc - t_obs) ** 2))),
            "error_rel_velocidad_%": float(100 * np.linalg.norm(v - v_true) / np.linalg.norm(v_true)),
            **anomaly_metrics(v, grid), "tiempo_s": float(seconds)}


def run_one(name):
    grid = Grid(); X, Z = grid.mesh(); v_true = true_velocity(X, Z)
    src, rec, t_obs = make_data(name, grid, v_true)
    fmm = lambda v, s: eikonal_fmm(v, grid, s)
    print(f"== {name}: {GEOMETRIES[name]}", flush=True)

    # convencional: barrido de lam, principio de discrepancia (mayor lam con RMS <= 1.05 x ruido)
    t0 = time.time()
    sweep = {}
    for lam in LAMS:
        v, hist = invert(t_obs, src, rec, fmm, grid.x, n_nodes=26, n_iter=8, lam=lam, verbose=False)
        sweep[lam] = (v, hist[-1])
        print(f"  conv lam {lam:5.2f} -> RMS {hist[-1]:.2f} ms", flush=True)
    ok = [l for l in LAMS if sweep[l][1] <= 1.05 * NOISE * 1e3]
    lam = max(ok) if ok else min(LAMS, key=lambda l: sweep[l][1])
    v_conv = sweep[lam][0]
    t_conv = (time.time() - t0) / len(LAMS)          # coste de una inversión
    conv = summarize(v_conv, fmm_times(v_conv, grid, src, rec), t_obs, v_true, grid, t_conv)
    conv["lam"] = lam

    t0 = time.time()
    v_net = train_pinn(src, rec, t_obs)
    t_pinn = time.time() - t0
    with torch.no_grad():
        v_pinn = v_net(torch.tensor(np.stack([X.ravel(), Z.ravel()], 1), dtype=torch.float32)).numpy().reshape(X.shape)
    pinn = summarize(v_pinn, fmm_times(v_pinn, grid, src, rec), t_obs, v_true, grid, t_pinn)

    # rayos en el modelo verdadero (cobertura)
    rays, density = [], np.zeros((26, 26))
    for s in src:
        T = fmm(v_true, s)
        rid, mid, ln = trace_rays(T, grid.x, s, rec, step=0.5 * grid.dx)
        for j in range(len(rec)):
            m = mid[rid == j]
            rays.append(m[::4])
        ij = np.clip((mid / (1 / 25.0)).round().astype(int), 0, 25)
        np.add.at(density, (ij[:, 1], ij[:, 0]), ln)
    cover = float(np.mean(density > 0.02))           # fracción de celdas con >20 m de rayo

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"geometria": name, "descripcion": GEOMETRIES[name], "cobertura_%": 100 * cover, "pinn": pinn, "convencional": conv}
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    np.savez_compressed(OUT / f"{name}.npz", v_pinn=v_pinn.astype(np.float32), v_conv=v_conv.astype(np.float32),
                        src=src, rec=rec, density=density,
                        rays=np.array([r for r in rays], dtype=object))
    print(json.dumps(res, indent=2, ensure_ascii=False))


def summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = Grid(); X, Z = grid.mesh(); v_true = true_velocity(X, Z)
    names = list(GEOMETRIES)
    res = {n: json.loads((OUT / f"{n}.json").read_text()) for n in names}
    titles = {"entre_pozos": "Entre pozos (actual)", "superficie": "Solo superficie", "vsp": "VSP", "cuatro_lados": "Cuatro lados"}

    fig, ax = plt.subplots(5, 4, figsize=(16, 20), constrained_layout=True)
    ext = [0, 1, 1, 0]; lo, hi = v_true.min(), v_true.max()
    for j, n in enumerate(names):
        d = np.load(OUT / f"{n}.npz", allow_pickle=True)
        a = ax[0, j]
        a.imshow(v_true, extent=ext, cmap="jet_r", vmin=lo, vmax=hi, alpha=0.35)
        for r in d["rays"][::2]:
            if len(r) > 1:
                a.plot(r[:, 0], r[:, 1], "k-", lw=0.3, alpha=0.5)
        a.plot(d["src"][:, 0], d["src"][:, 1], "r*", ms=9, clip_on=False)
        a.plot(d["rec"][:, 0], d["rec"][:, 1], "bv", ms=4, clip_on=False)
        a.set_title(f"{titles[n]}\nrayos (cobertura {res[n]['cobertura_%']:.0f} %)")
        for i, (key, lab) in enumerate([("v_pinn", "PINN"), ("v_conv", "Convencional")]):
            m = res[n]["pinn" if key == "v_pinn" else "convencional"]
            ax[1 + i, j].imshow(d[key], extent=ext, cmap="jet_r", vmin=lo, vmax=hi)
            ax[1 + i, j].set_title(f"{lab}: error {m['error_rel_velocidad_%']:.2f} %, "
                                   f"rápida {m['anomalia_rapida_amplitud_recuperada_%']:.0f} %", fontsize=10)
            ax[3 + i, j].imshow(d[key] - v_true, extent=ext, cmap="seismic", vmin=-0.5, vmax=0.5)
            ax[3 + i, j].set_title(f"Error {lab}", fontsize=10)
    for a in ax.flat:
        a.set(xlim=(0, 1), ylim=(1, 0), xlabel="x (km)", ylabel="z (km)")
    fig.colorbar(ax[1, 3].images[0], ax=ax[1:3, 3], shrink=0.7, label="km/s")
    fig.colorbar(ax[3, 3].images[0], ax=ax[3:, 3], shrink=0.7, label="km/s")
    fig.savefig(OUT / "comparacion_geometrias.png", dpi=95)

    lines = ["| Geometría | Cobertura de rayos | Método | RMS tiempos (ms) | Error velocidad (%) | Anomalía lenta (%) | Anomalía rápida (%) |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        for key, lab in [("pinn", "PINN"), ("convencional", f"Convencional (λ = {res[n]['convencional']['lam']})")]:
            m = res[n][key]
            lines.append(f"| {titles[n] if key == 'pinn' else ''} | {str(round(res[n]['cobertura_%'])) + ' %' if key == 'pinn' else ''} | {lab} | "
                         f"{m['rms_tiempos_ms']:.2f} | {m['error_rel_velocidad_%']:.2f} | "
                         f"{m['anomalia_lenta_amplitud_recuperada_%']:.0f} | {m['anomalia_rapida_amplitud_recuperada_%']:.0f} |")
    table = "\n".join(lines)
    (OUT / "tabla.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geom", choices=list(GEOMETRIES))
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        summary()
    elif a.geom:
        run_one(a.geom)
    else:
        p.error("usa --geom NOMBRE o --summary")
