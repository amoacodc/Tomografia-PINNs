"""Tomografía sísmica de tiempos de viaje con PINN — script principal.

Uso:
    python run_tomography.py                 # configuración por defecto
    python run_tomography.py --iters 8000 --noise 0.002
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_tomo.forward import Grid, crosswell_geometry, traveltimes, true_velocity
from pinn_tomo.pinn import TauNet, VelocityNet, eikonal_residual, traveltime, velocity_roughness


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-colloc", type=int, default=3000, help="puntos de colocación por iteración")
    p.add_argument("--w-pde", type=float, default=100.0)
    p.add_argument("--w-data", type=float, default=1.0)
    p.add_argument("--w-reg", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=1e-3, help="desv. estándar del ruido en los tiempos (s)")
    p.add_argument("--vmin", type=float, default=1.0)
    p.add_argument("--vmax", type=float, default=4.0)
    p.add_argument("--v-fourier", type=int, default=0, help="nº de características de Fourier en v_net (0 = sin)")
    p.add_argument("--v-sigma", type=float, default=2.0, help="escala de frecuencias de Fourier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs")
    return p.parse_args()


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- datos sintéticos ----------------
    grid = Grid()
    X, Z = grid.mesh()
    v_true = true_velocity(X, Z)
    src, rec = crosswell_geometry()
    t_clean, _ = traveltimes(v_true, grid, src, rec)
    t_obs = t_clean + a.noise * rng.standard_normal(t_clean.shape)
    print(f"{len(src)} fuentes x {len(rec)} receptores = {t_obs.size} tiempos; "
          f"T en [{t_obs.min():.3f}, {t_obs.max():.3f}] s")

    # pares (fuente, receptor) como tensores
    S = torch.tensor(np.repeat(src, len(rec), axis=0), dtype=torch.float32)
    R = torch.tensor(np.tile(rec, (len(src), 1)), dtype=torch.float32)
    T_obs = torch.tensor(t_obs.reshape(-1, 1), dtype=torch.float32)
    src_t = torch.tensor(src, dtype=torch.float32)
    t_scale = float(T_obs.abs().mean())

    # ---------------- redes ----------------
    v_net = VelocityNet(a.vmin, a.vmax, n_fourier=a.v_fourier, sigma=a.v_sigma)
    tau_net = TauNet(s_ref=2.0 / (a.vmin + a.vmax))
    opt = torch.optim.Adam(list(v_net.parameters()) + list(tau_net.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters, eta_min=a.lr * 0.02)

    hist = {"it": [], "pde": [], "data": [], "rms_ms": []}
    t_start = time.time()
    for it in range(a.iters + 1):
        xz = torch.rand(a.n_colloc, 2)
        sxz = src_t[torch.randint(len(src_t), (a.n_colloc,))]
        l_pde = (eikonal_residual(tau_net, v_net, xz, sxz) ** 2).mean()
        t_pred = traveltime(tau_net, R, S)
        l_data = (((t_pred - T_obs) / t_scale) ** 2).mean() * 1e3
        l_reg = velocity_roughness(v_net, torch.rand(1000, 2))
        loss = a.w_pde * l_pde + a.w_data * l_data + a.w_reg * l_reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 200 == 0:
            rms = float(torch.sqrt(((t_pred.detach() - T_obs) ** 2).mean())) * 1e3
            hist["it"].append(it); hist["pde"].append(float(l_pde.detach()))
            hist["data"].append(float(l_data.detach())); hist["rms_ms"].append(rms)
            print(f"it {it:6d} | pde {l_pde:.2e} | data {l_data:.2e} | rms {rms:6.2f} ms "
                  f"| {time.time() - t_start:6.1f} s")

    # ---------------- evaluación ----------------
    with torch.no_grad():
        xz_grid = torch.tensor(np.stack([X.ravel(), Z.ravel()], 1), dtype=torch.float32)
        v_inv = v_net(xz_grid).numpy().reshape(X.shape)
    # validación independiente: tiempos de FMM en el modelo invertido
    t_fmm_inv, _ = traveltimes(v_inv, grid, src, rec)
    t_fmm_init, _ = traveltimes(np.full_like(v_true, (a.vmin + a.vmax) / 2), grid, src, rec)
    metrics = {
        "rmse_velocidad_km_s": float(np.sqrt(np.mean((v_inv - v_true) ** 2))),
        "error_rel_velocidad_%": float(100 * np.linalg.norm(v_inv - v_true) / np.linalg.norm(v_true)),
        "rms_tiempos_modelo_inicial_ms": float(1e3 * np.sqrt(np.mean((t_fmm_init - t_obs) ** 2))),
        "rms_tiempos_modelo_invertido_ms (FMM)": float(1e3 * np.sqrt(np.mean((t_fmm_inv - t_obs) ** 2))),
        "ruido_ms": a.noise * 1e3,
        "tiempo_entrenamiento_s": time.time() - t_start,
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    np.savez(out / "results.npz", x=grid.x, z=grid.z, v_true=v_true, v_inv=v_inv,
             src=src, rec=rec, t_obs=t_obs, t_fmm_inv=t_fmm_inv, **{f"hist_{k}": v for k, v in hist.items()})
    torch.save({"v_net": v_net.state_dict(), "tau_net": tau_net.state_dict()}, out / "model.pt")
    plot(grid, v_true, v_inv, src, rec, t_obs, t_fmm_inv, hist, out)


def plot(grid, v_true, v_inv, src, rec, t_obs, t_fmm_inv, hist, out):
    ext = [grid.xmin, grid.xmax, grid.zmax, grid.zmin]
    vmin, vmax = v_true.min(), v_true.max()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for a_, data, title in [(ax[0], v_true, "Velocidad verdadera"), (ax[1], v_inv, "Velocidad invertida (PINN)")]:
        im = a_.imshow(data, extent=ext, cmap="jet_r", vmin=vmin, vmax=vmax)
        a_.plot(src[:, 0], src[:, 1], "r*", ms=9, label="fuentes")
        a_.plot(rec[:, 0], rec[:, 1], "kv", ms=4, label="receptores")
        a_.set(title=title, xlabel="x (km)", ylabel="z (km)")
        fig.colorbar(im, ax=a_, label="km/s")
    ax[0].legend(loc="lower right", fontsize=8)
    err = v_inv - v_true
    m = np.abs(err).max()
    im = ax[2].imshow(err, extent=ext, cmap="seismic", vmin=-m, vmax=m)
    ax[2].set(title="Error (invertida − verdadera)", xlabel="x (km)", ylabel="z (km)")
    fig.colorbar(im, ax=ax[2], label="km/s")
    fig.savefig(out / "velocidad.png", dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    ax[0].semilogy(hist["it"], hist["pde"], label="PDE (eikonal)")
    ax[0].semilogy(hist["it"], hist["data"], label="datos")
    ax[0].set(title="Pérdidas", xlabel="iteración"); ax[0].legend()
    ax[1].plot(hist["it"], hist["rms_ms"]); ax[1].set(title="RMS tiempos (PINN)", xlabel="iteración", ylabel="ms")
    ix = np.argmin(np.abs(grid.x - 0.55))
    ax[2].plot(v_true[:, ix], grid.z, "k", label="verdadera")
    ax[2].plot(v_inv[:, ix], grid.z, "r--", label="PINN")
    ax[2].invert_yaxis(); ax[2].set(title="Perfil en x = 0.55 km", xlabel="v (km/s)", ylabel="z (km)"); ax[2].legend()
    fig.savefig(out / "entrenamiento.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot(t_obs.ravel(), t_fmm_inv.ravel(), ".", ms=3)
    lim = [t_obs.min(), t_obs.max()]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(title="Tiempos: observados vs. FMM en modelo PINN", xlabel="T obs (s)", ylabel="T calc (s)")
    fig.savefig(out / "ajuste_tiempos.png", dpi=130)


if __name__ == "__main__":
    main()
