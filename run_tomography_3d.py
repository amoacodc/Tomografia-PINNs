"""Tomografía sísmica de tiempos de viaje 3D con PINN.

Genera datos sintéticos 3D (o los lee de data/3d/), invierte la velocidad con una PINN
eikonal factorizada y valida el resultado recalculando los tiempos con Fast Marching 3D.

Uso:
    python run_tomography_3d.py
    python run_tomography_3d.py --iters 15000 --v-width 64
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

from pinn_tomo.forward3d import Grid3D, anomaly_metrics, geometry_3d, traveltimes, true_velocity
from pinn_tomo.pinn import TauNet, VelocityNet, eikonal_residual, traveltime, velocity_roughness


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-colloc", type=int, default=4000)
    p.add_argument("--w-pde", type=float, default=100.0)
    p.add_argument("--w-data", type=float, default=1.0)
    p.add_argument("--w-reg", type=float, default=0.0)
    p.add_argument("--v-width", type=int, default=48)
    p.add_argument("--tau-width", type=int, default=64)
    p.add_argument("--noise", type=float, default=1e-3, help="desv. estándar del ruido (s)")
    p.add_argument("--n-grid", type=int, default=161, help="malla FMM para datos y validación")
    p.add_argument("--vmin", type=float, default=1.0)
    p.add_argument("--vmax", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=str, default="data/3d")
    p.add_argument("--out", type=str, default="outputs_3d")
    return p.parse_args()


def make_or_load_data(a, grid, v_true):
    d = Path(a.data)
    f = d / "tiempos.csv"
    if f.exists():
        print(f"Leyendo datos de {f}")
        tab = np.loadtxt(f, delimiter=",", skiprows=1)
        src = np.loadtxt(d / "fuentes.csv", delimiter=",", skiprows=1)[:, 1:]
        rec = np.loadtxt(d / "receptores.csv", delimiter=",", skiprows=1)[:, 1:]
        t_obs = np.zeros((len(src), len(rec)))
        t_obs[tab[:, 0].astype(int), tab[:, 4].astype(int)] = tab[:, 8]
        return src, rec, t_obs
    print("Generando datos sintéticos 3D con FMM...")
    rng = np.random.default_rng(a.seed)
    src, rec = geometry_3d()
    t_clean = traveltimes(v_true, grid, src, rec)
    t_obs = t_clean + a.noise * rng.standard_normal(t_clean.shape)
    d.mkdir(parents=True, exist_ok=True)
    fmt3 = ["%d", "%.4f", "%.4f", "%.4f"]
    np.savetxt(d / "fuentes.csv", np.c_[np.arange(len(src)), src], delimiter=",",
               header="id,x_km,y_km,z_km", comments="", fmt=fmt3)
    np.savetxt(d / "receptores.csv", np.c_[np.arange(len(rec)), rec], delimiter=",",
               header="id,x_km,y_km,z_km", comments="", fmt=fmt3)
    rows = [(i, *src[i], j, *rec[j], t_obs[i, j], t_clean[i, j])
            for i in range(len(src)) for j in range(len(rec))]
    np.savetxt(d / "tiempos.csv", np.array(rows), delimiter=",",
               header="id_fuente,xs_km,ys_km,zs_km,id_receptor,xr_km,yr_km,zr_km,t_obs_s,t_sin_ruido_s",
               comments="", fmt=fmt3 + fmt3 + ["%.6f", "%.6f"])
    (d / "info.json").write_text(json.dumps({
        "descripcion": f"Datos sintéticos 3D: tiempos de primera llegada con Fast Marching "
                       f"(scikit-fmm, 2º orden, malla {grid.n}^3) sobre el modelo verdadero + ruido gaussiano.",
        "ruido_desv_std_s": a.noise, "semilla": a.seed,
        "n_fuentes": len(src), "n_receptores": len(rec), "n_tiempos": int(t_obs.size),
        "geometria": "fuentes en 2 pozos (x=0.02; y=0.25, 0.75); receptores en 3 pozos "
                     "(x=0.98,y=0.25 / x=0.98,y=0.75 / x=0.50,y=0.98) y malla 8x8 en superficie",
        "modelo": "v = 2.0 + 1.0 z - 0.6 exp(-|r-(0.55,0.50,0.45)|^2/(2*0.12^2)) "
                  "+ 0.5 exp(-|r-(0.35,0.35,0.75)|^2/(2*0.10^2))  [km/s, km]; "
                  "ver pinn_tomo/forward3d.py::true_velocity",
    }, indent=2, ensure_ascii=False))
    print(f"Datos guardados en {d}/")
    return src, rec, t_obs


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    grid = Grid3D(n=a.n_grid)
    X, Y, Z = grid.mesh()
    v_true = true_velocity(X, Y, Z)
    src, rec, t_obs = make_or_load_data(a, grid, v_true)
    print(f"{len(src)} fuentes x {len(rec)} receptores = {t_obs.size} tiempos")

    S = torch.tensor(np.repeat(src, len(rec), axis=0), dtype=torch.float32)
    R = torch.tensor(np.tile(rec, (len(src), 1)), dtype=torch.float32)
    T_obs = torch.tensor(t_obs.reshape(-1, 1), dtype=torch.float32)
    src_t = torch.tensor(src, dtype=torch.float32)
    t_scale = float(T_obs.abs().mean())

    v_net = VelocityNet(a.vmin, a.vmax, width=a.v_width, dim=3)
    tau_net = TauNet(s_ref=2.0 / (a.vmin + a.vmax), width=a.tau_width, dim=3)
    opt = torch.optim.Adam(list(v_net.parameters()) + list(tau_net.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters, eta_min=a.lr * 0.02)

    hist = {"it": [], "pde": [], "data": [], "rms_ms": []}
    t0 = time.time()
    for it in range(a.iters + 1):
        xyz = torch.rand(a.n_colloc, 3)
        sxyz = src_t[torch.randint(len(src_t), (a.n_colloc,))]
        l_pde = (eikonal_residual(tau_net, v_net, xyz, sxyz) ** 2).mean()
        t_pred = traveltime(tau_net, R, S)
        l_data = (((t_pred - T_obs) / t_scale) ** 2).mean() * 1e3
        loss = a.w_pde * l_pde + a.w_data * l_data
        if a.w_reg > 0:
            loss = loss + a.w_reg * velocity_roughness(v_net, torch.rand(1000, 3))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 250 == 0:
            rms = float(torch.sqrt(((t_pred.detach() - T_obs) ** 2).mean())) * 1e3
            hist["it"].append(it); hist["pde"].append(float(l_pde.detach()))
            hist["data"].append(float(l_data.detach())); hist["rms_ms"].append(rms)
            print(f"it {it:6d} | pde {l_pde.item():.2e} | data {l_data.item():.2e} | rms {rms:6.2f} ms "
                  f"| {time.time() - t0:6.1f} s", flush=True)
    train_time = time.time() - t0

    with torch.no_grad():
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1), dtype=torch.float32)
        v_inv = torch.cat([v_net(c) for c in pts.split(200000)]).numpy().reshape(X.shape)
    print("Validando con FMM 3D...")
    t_fmm_inv = traveltimes(v_inv, grid, src, rec)
    t_fmm_init = traveltimes(np.full_like(v_true, (a.vmin + a.vmax) / 2), grid, src, rec)
    metrics = {
        "rmse_velocidad_km_s": float(np.sqrt(np.mean((v_inv - v_true) ** 2))),
        "error_rel_velocidad_%": float(100 * np.linalg.norm(v_inv - v_true) / np.linalg.norm(v_true)),
        "rms_tiempos_modelo_inicial_ms": float(1e3 * np.sqrt(np.mean((t_fmm_init - t_obs) ** 2))),
        "rms_tiempos_modelo_invertido_ms (FMM)": float(1e3 * np.sqrt(np.mean((t_fmm_inv - t_obs) ** 2))),
        "ruido_ms": a.noise * 1e3,
        **anomaly_metrics(v_inv, grid),
        "tiempo_entrenamiento_s": train_time,
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    np.savez_compressed(out / "results.npz", axis_km=grid.axis, v_true=v_true.astype(np.float32),
                        v_inv=v_inv.astype(np.float32), src=src, rec=rec, t_obs=t_obs, t_fmm_inv=t_fmm_inv)
    torch.save({"v_net": v_net.state_dict(), "tau_net": tau_net.state_dict()}, out / "model.pt")
    plot(grid, v_true, v_inv, src, rec, hist, out)


def plot(grid, v_true, v_inv, src, rec, hist, out):
    a = grid.axis
    idx = lambda c: int(np.argmin(np.abs(a - c)))
    vmin, vmax = v_true.min(), v_true.max()
    # cortes que pasan por las dos anomalías
    cuts = [("z = 0.45 km (horizontal)", lambda v: v[idx(0.45)], "x", "y"),
            ("z = 0.75 km (horizontal)", lambda v: v[idx(0.75)], "x", "y"),
            ("y = 0.50 km (vertical)", lambda v: v[:, idx(0.50), :], "x", "z"),
            ("y = 0.35 km (vertical)", lambda v: v[:, idx(0.35), :], "x", "z")]
    fig, ax = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
    for j, (title, f, xl, yl) in enumerate(cuts):
        horiz = yl == "y"
        ext = [0, grid.L, 0, grid.L] if horiz else [0, grid.L, grid.L, 0]
        origin = "lower" if horiz else "upper"
        err = f(v_inv) - f(v_true)
        for i, (data, lab, cmap, lims) in enumerate([
                (f(v_true), "Verdadera", "jet_r", (vmin, vmax)),
                (f(v_inv), "PINN", "jet_r", (vmin, vmax)),
                (err, "Error", "seismic", (-0.4, 0.4))]):
            im = ax[i, j].imshow(data, extent=ext, origin=origin, cmap=cmap, vmin=lims[0], vmax=lims[1])
            ax[i, j].set(title=f"{lab} — {title}", xlabel=f"{xl} (km)", ylabel=f"{yl} (km)")
            if i < 2 and horiz and j == 0:
                ax[i, j].plot(src[:, 0], src[:, 1], "r*", ms=8)
                ax[i, j].plot(rec[:, 0], rec[:, 1], "kv", ms=3)
        fig.colorbar(im, ax=ax[2, j], label="km/s", shrink=0.8)
    fig.colorbar(ax[0, 3].images[0], ax=ax[:2, 3], label="km/s", shrink=0.6)
    fig.savefig(out / "velocidad_cortes.png", dpi=110)

    fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
    a3 = fig.add_subplot(1, 3, 1, projection="3d")
    a3.scatter(src[:, 0], src[:, 1], src[:, 2], c="r", marker="*", s=60, label="fuentes")
    a3.scatter(rec[:, 0], rec[:, 1], rec[:, 2], c="k", marker="v", s=10, label="receptores")
    a3.set(xlabel="x", ylabel="y", zlabel="z", title="Geometría 3D (km)"); a3.invert_zaxis(); a3.legend()
    a1 = fig.add_subplot(1, 3, 2)
    a1.semilogy(hist["it"], hist["pde"], label="PDE"); a1.semilogy(hist["it"], hist["data"], label="datos")
    a1.set(title="Pérdidas", xlabel="iteración"); a1.legend()
    a2 = fig.add_subplot(1, 3, 3)
    ix, iy = idx(0.55), idx(0.50)
    a2.plot(v_true[:, iy, ix], a, "k", label="verdadera"); a2.plot(v_inv[:, iy, ix], a, "r--", label="PINN")
    a2.invert_yaxis(); a2.set(title="Perfil en (x, y) = (0.55, 0.50)", xlabel="v (km/s)", ylabel="z (km)"); a2.legend()
    fig.savefig(out / "entrenamiento.png", dpi=110)


if __name__ == "__main__":
    main()
