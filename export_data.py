"""Exporta los datos sintéticos usados en la inversión a data/.

Reproduce exactamente los datos de run_tomography.py (misma semilla y mismo ruido).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pinn_tomo.forward import Grid, crosswell_geometry, traveltimes, true_velocity

p = argparse.ArgumentParser()
p.add_argument("--noise", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=str, default="data")
a = p.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

rng = np.random.default_rng(a.seed)
grid = Grid()
X, Z = grid.mesh()
v_true = true_velocity(X, Z)
src, rec = crosswell_geometry()
t_clean, _ = traveltimes(v_true, grid, src, rec)
t_obs = t_clean + a.noise * rng.standard_normal(t_clean.shape)

np.savetxt(out / "fuentes.csv", np.c_[np.arange(len(src)), src], delimiter=",",
           header="id,x_km,z_km", comments="", fmt=["%d", "%.4f", "%.4f"])
np.savetxt(out / "receptores.csv", np.c_[np.arange(len(rec)), rec], delimiter=",",
           header="id,x_km,z_km", comments="", fmt=["%d", "%.4f", "%.4f"])
rows = [(i, *src[i], j, *rec[j], t_obs[i, j], t_clean[i, j])
        for i in range(len(src)) for j in range(len(rec))]
np.savetxt(out / "tiempos.csv", np.array(rows), delimiter=",",
           header="id_fuente,xs_km,zs_km,id_receptor,xr_km,zr_km,t_obs_s,t_sin_ruido_s",
           comments="", fmt=["%d", "%.4f", "%.4f", "%d", "%.4f", "%.4f", "%.6f", "%.6f"])
np.savez_compressed(out / "modelo_verdadero.npz", x_km=grid.x, z_km=grid.z, v_km_s=v_true)
(out / "info.json").write_text(json.dumps({
    "descripcion": "Datos sintéticos: tiempos de primera llegada calculados con Fast Marching "
                   "(scikit-fmm, 2º orden, malla 201x201) sobre el modelo verdadero, más ruido gaussiano.",
    "ruido_desv_std_s": a.noise, "semilla": a.seed,
    "n_fuentes": len(src), "n_receptores": len(rec), "n_tiempos": int(t_obs.size),
    "modelo": "v = 2.0 + 1.0 z - 0.6 exp(-((x-0.55)^2+(z-0.45)^2)/(2*0.10^2)) "
              "+ 0.5 exp(-((x-0.30)^2+(z-0.78)^2)/(2*0.08^2))  [km/s, x y z en km]",
}, indent=2, ensure_ascii=False))
print(f"Guardado en {out}/: {t_obs.size} tiempos")
