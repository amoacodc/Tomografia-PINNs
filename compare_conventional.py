"""Comparación PINN vs. tomografía convencional de rayos curvos (2D o 3D).

La inversión convencional (pinn_tomo/conventional.py) usa los mismos datos, el mismo modelo
inicial (2.5 km/s) y el mismo solver directo (FMM) que la validación de la PINN. Su
regularización (lam) se eligió con el principio de discrepancia: el mayor lam cuyo ajuste
queda al nivel del ruido (~1 ms), sin mirar el modelo verdadero.

Requiere los resultados de la PINN (outputs/results.npz o outputs_3d/results.npz).
    python compare_conventional.py --dim 2
    python compare_conventional.py --dim 3
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinn_tomo.conventional import invert

DEFAULT_LAM = {2: 0.2, 3: 0.1}   # elegidos con el principio de discrepancia


def load_2d():
    from pinn_tomo.forward import Grid, anomaly_metrics, eikonal_fmm, true_velocity
    g = Grid(); X, Z = g.mesh()
    tab = np.loadtxt("data/tiempos.csv", delimiter=",", skiprows=1)
    src = np.loadtxt("data/fuentes.csv", delimiter=",", skiprows=1)[:, 1:]
    rec = np.loadtxt("data/receptores.csv", delimiter=",", skiprows=1)[:, 1:]
    t = np.zeros((len(src), len(rec))); t[tab[:, 0].astype(int), tab[:, 3].astype(int)] = tab[:, 6]
    pinn = np.load("outputs/results.npz")
    pinn_m = json.load(open("outputs/metrics.json"))
    return dict(grid=g, axis=g.x, v_true=true_velocity(X, Z), src=src, rec=rec, t_obs=t,
                fmm=lambda v, s: eikonal_fmm(v, g, s), metrics=anomaly_metrics,
                v_pinn=pinn["v_inv"], t_pinn=pinn["t_fmm_inv"], pinn_time=pinn_m["tiempo_entrenamiento_s"],
                nodes=26, out=Path("outputs"))


def load_3d():
    from pinn_tomo.forward3d import Grid3D, anomaly_metrics, eikonal_fmm, true_velocity
    g = Grid3D(n=161); X, Y, Z = g.mesh()
    tab = np.loadtxt("data/3d/tiempos.csv", delimiter=",", skiprows=1)
    src = np.loadtxt("data/3d/fuentes.csv", delimiter=",", skiprows=1)[:, 1:]
    rec = np.loadtxt("data/3d/receptores.csv", delimiter=",", skiprows=1)[:, 1:]
    t = np.zeros((len(src), len(rec))); t[tab[:, 0].astype(int), tab[:, 4].astype(int)] = tab[:, 8]
    pinn = np.load("outputs_3d/results.npz")
    pinn_m = json.load(open("outputs_3d/metrics.json"))
    return dict(grid=g, axis=g.axis, v_true=true_velocity(X, Y, Z), src=src, rec=rec, t_obs=t,
                fmm=lambda v, s: eikonal_fmm(v, g, s), metrics=anomaly_metrics,
                v_pinn=pinn["v_inv"].astype(float), t_pinn=pinn["t_fmm_inv"], pinn_time=pinn_m["tiempo_entrenamiento_s"],
                nodes=21, out=Path("outputs_3d"))


def summarize(name, v, t_calc, d, seconds):
    vt = d["v_true"]
    m = {"metodo": name,
         "rms_tiempos_ms (FMM)": float(1e3 * np.sqrt(np.mean((t_calc - d["t_obs"]) ** 2))),
         "error_rel_velocidad_%": float(100 * np.linalg.norm(v - vt) / np.linalg.norm(vt)),
         "rmse_velocidad_km_s": float(np.sqrt(np.mean((v - vt) ** 2))),
         **d["metrics"](v, d["grid"]),
         "tiempo_s": float(seconds)}
    return m


def plot_2d(d, v_conv, out):
    vt, vp = d["v_true"], d["v_pinn"]
    ext = [0, 1, 1, 0]; lo, hi = vt.min(), vt.max()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for a, data, title in [(ax[0, 0], vt, "Verdadera"), (ax[0, 1], vp, "PINN"), (ax[0, 2], v_conv, "Convencional (rayos curvos)")]:
        im = a.imshow(data, extent=ext, cmap="jet_r", vmin=lo, vmax=hi); a.set_title(title)
        a.plot(d["src"][:, 0], d["src"][:, 1], "r*", ms=7); a.plot(d["rec"][:, 0], d["rec"][:, 1], "kv", ms=3)
    fig.colorbar(im, ax=ax[0, :], shrink=0.8, label="km/s")
    for a, data, title in [(ax[1, 0], vp - vt, "Error PINN"), (ax[1, 1], v_conv - vt, "Error convencional")]:
        im2 = a.imshow(data, extent=ext, cmap="seismic", vmin=-0.4, vmax=0.4); a.set_title(title)
    fig.colorbar(im2, ax=ax[1, :2], shrink=0.8, label="km/s")
    z = d["axis"]
    for xc, ls in [(0.55, "-"), (0.30, "--")]:
        ix = int(np.argmin(np.abs(d["axis"] - xc)))
        ax[1, 2].plot(vt[:, ix], z, "k" + ls, label=f"verdadera x={xc}")
        ax[1, 2].plot(vp[:, ix], z, "r" + ls, label=f"PINN x={xc}")
        ax[1, 2].plot(v_conv[:, ix], z, "b" + ls, label=f"convencional x={xc}")
    ax[1, 2].invert_yaxis(); ax[1, 2].set(title="Perfiles verticales", xlabel="v (km/s)", ylabel="z (km)")
    ax[1, 2].legend(fontsize=7)
    for a in ax.flat[:5]:
        a.set(xlabel="x (km)", ylabel="z (km)")
    fig.savefig(out, dpi=120)


def plot_3d(d, v_conv, out):
    vt, vp = d["v_true"], d["v_pinn"]; a_km = d["axis"]
    idx = lambda c: int(np.argmin(np.abs(a_km - c)))
    cuts = [("z = 0.45 km", lambda v: v[idx(0.45)], True), ("z = 0.75 km", lambda v: v[idx(0.75)], True),
            ("y = 0.50 km", lambda v: v[:, idx(0.50), :], False), ("y = 0.35 km", lambda v: v[:, idx(0.35), :], False)]
    rows = [("Verdadera", lambda f: f(vt), "jet_r"), ("PINN", lambda f: f(vp), "jet_r"),
            ("Convencional", lambda f: f(v_conv), "jet_r"),
            ("Error PINN", lambda f: f(vp) - f(vt), "seismic"), ("Error conv.", lambda f: f(v_conv) - f(vt), "seismic")]
    fig, ax = plt.subplots(5, 4, figsize=(16, 19), constrained_layout=True)
    for j, (name, f, horiz) in enumerate(cuts):
        ext, org = ([0, 1, 0, 1], "lower") if horiz else ([0, 1, 1, 0], "upper")
        for i, (lab, g, cmap) in enumerate(rows):
            lim = (-0.4, 0.4) if cmap == "seismic" else (vt.min(), vt.max())
            ax[i, j].imshow(g(f), extent=ext, origin=org, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax[i, j].set_title(f"{lab} — {name}", fontsize=10)
            ax[i, j].set(xlabel="x (km)", ylabel="y (km)" if horiz else "z (km)")
    fig.colorbar(ax[0, 3].images[0], ax=ax[:3, 3], shrink=0.5, label="km/s")
    fig.colorbar(ax[3, 3].images[0], ax=ax[3:, 3], shrink=0.6, label="km/s")
    fig.savefig(out, dpi=100)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)
    p.add_argument("--lam", type=float, default=None, help="regularización (por defecto la calibrada)")
    p.add_argument("--iters", type=int, default=8)
    a = p.parse_args()
    d = load_2d() if a.dim == 2 else load_3d()
    lam = a.lam if a.lam is not None else DEFAULT_LAM[a.dim]

    print(f"Inversión convencional {a.dim}D (lam = {lam}, {d['nodes']} nodos por eje)")
    t0 = time.time()
    v_conv, hist = invert(d["t_obs"], d["src"], d["rec"], d["fmm"], d["axis"],
                          n_nodes=d["nodes"], n_iter=a.iters, lam=lam)
    t_conv = time.time() - t0
    # tiempos FMM del modelo convencional final (misma malla que la validación de la PINN)
    from scipy.interpolate import RegularGridInterpolator
    t_calc = np.stack([RegularGridInterpolator((d["axis"],) * a.dim, d["fmm"](v_conv, s))(d["rec"][:, ::-1])
                       for s in d["src"]])

    res = [summarize("PINN", d["v_pinn"], d["t_pinn"], d, d["pinn_time"]),
           summarize("Convencional (rayos curvos)", v_conv, t_calc, d, t_conv)]
    res[1]["lam"] = lam; res[1]["rms_por_iteracion_ms"] = hist
    out = d["out"]
    (out / "comparacion.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    np.save(out / "v_convencional.npy", v_conv.astype(np.float32))
    (plot_2d if a.dim == 2 else plot_3d)(d, v_conv, out / "comparacion.png")

    keys = [k for k in res[0] if k != "metodo"]
    print(f"\n{'métrica':42s} {'PINN':>10s} {'convencional':>13s}")
    for k in keys:
        print(f"{k:42s} {res[0][k]:10.3f} {res[1][k]:13.3f}")
    print(f"\nGuardado {out}/comparacion.json y {out}/comparacion.png")


if __name__ == "__main__":
    main()
