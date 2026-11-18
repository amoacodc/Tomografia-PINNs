"""Modelo sintético y problema directo (ecuación eikonal resuelta con Fast Marching).

Se usa únicamente para generar los datos "observados" y para validar de forma
independiente el modelo de velocidad invertido por la PINN.
"""
from dataclasses import dataclass

import numpy as np
import skfmm
from scipy.interpolate import RegularGridInterpolator


@dataclass
class Grid:
    xmin: float = 0.0
    xmax: float = 1.0   # km
    zmin: float = 0.0
    zmax: float = 1.0   # km
    n: int = 201

    @property
    def x(self):
        return np.linspace(self.xmin, self.xmax, self.n)

    @property
    def z(self):
        return np.linspace(self.zmin, self.zmax, self.n)

    @property
    def dx(self):
        return (self.xmax - self.xmin) / (self.n - 1)

    def mesh(self):
        return np.meshgrid(self.x, self.z, indexing="xy")  # X[iz, ix], Z[iz, ix]


def true_velocity(x, z):
    """Velocidad verdadera (km/s): gradiente vertical + anomalía lenta + anomalía rápida."""
    v = 2.0 + 1.0 * z
    v = v - 0.6 * np.exp(-((x - 0.55) ** 2 + (z - 0.45) ** 2) / (2 * 0.10 ** 2))
    v = v + 0.5 * np.exp(-((x - 0.30) ** 2 + (z - 0.78) ** 2) / (2 * 0.08 ** 2))
    return v


ANOMALIES = {"lenta": ((0.55, 0.45), -0.6, 0.10), "rapida": ((0.30, 0.78), 0.5, 0.08)}


def anomaly_metrics(v_inv, grid):
    """Amplitud recuperada en el centro de cada anomalía (% de la verdadera, respecto al
    fondo 2 + z) y RMSE dentro de un círculo de radio 2 sigma."""
    X, Z = grid.mesh()
    v_true = true_velocity(X, Z)
    out = {}
    for name, (c, amp, sig) in ANOMALIES.items():
        ix, iz = (int(round(ci / grid.dx)) for ci in c)
        mask = (X - c[0]) ** 2 + (Z - c[1]) ** 2 < (2 * sig) ** 2
        out[f"anomalia_{name}_amplitud_recuperada_%"] = float(100 * (v_inv[iz, ix] - (2.0 + c[1])) / amp)
        out[f"anomalia_{name}_rmse_km_s"] = float(np.sqrt(np.mean((v_inv[mask] - v_true[mask]) ** 2)))
    return out


def crosswell_geometry(n_src=12, n_rec_well=31, n_rec_surf=21):
    """Fuentes en el pozo izquierdo; receptores en el pozo derecho y en superficie."""
    src = np.stack([np.full(n_src, 0.02), np.linspace(0.05, 0.95, n_src)], axis=1)
    rec_w = np.stack([np.full(n_rec_well, 0.98), np.linspace(0.02, 0.98, n_rec_well)], axis=1)
    rec_s = np.stack([np.linspace(0.10, 0.90, n_rec_surf), np.full(n_rec_surf, 0.0)], axis=1)
    return src, np.concatenate([rec_w, rec_s], axis=0)


GEOMETRIES = {
    "entre_pozos": "Entre pozos (actual): fuentes en el pozo izquierdo; receptores en el pozo derecho y en superficie",
    "superficie": "Solo superficie: fuentes y receptores en z = 0",
    "vsp": "VSP: fuentes en superficie; receptores en el pozo derecho y en superficie",
    "cuatro_lados": "Cuatro lados: fuentes en el pozo izquierdo y en superficie; receptores en el pozo derecho y en el fondo",
}


def geometry(name):
    """Geometrías de adquisición con el mismo número de datos (12 fuentes x 52 receptores)."""
    def line(x, z):
        return np.stack([np.broadcast_to(x, np.shape(z if np.ndim(z) else x)),
                         np.broadcast_to(z, np.shape(x if np.ndim(x) else z))], axis=1).astype(float)
    if name == "entre_pozos":
        return crosswell_geometry()
    if name == "superficie":
        src = line(np.linspace(0.04, 0.96, 12), 0.0)
        rec = line(np.linspace(0.02, 0.98, 52), 0.0)
    elif name == "vsp":
        src = line(np.linspace(0.04, 0.90, 12), 0.0)
        rec = np.concatenate([line(0.98, np.linspace(0.02, 0.98, 31)), line(np.linspace(0.10, 0.90, 21), 0.0)])
    elif name == "cuatro_lados":
        src = np.concatenate([line(0.02, np.linspace(0.10, 0.90, 6)), line(np.linspace(0.15, 0.85, 6), 0.0)])
        rec = np.concatenate([line(0.98, np.linspace(0.02, 0.98, 26)), line(np.linspace(0.04, 0.96, 26), 0.98)])
    else:
        raise ValueError(name)
    return src, rec


def eikonal_fmm(v_grid, grid: Grid, src_xz):
    """Tiempo de viaje T(x, z) para una fuente puntual con Fast Marching de 2º orden."""
    X, Z = grid.mesh()
    r0 = 2.0 * grid.dx
    phi = np.sqrt((X - src_xz[0]) ** 2 + (Z - src_xz[1]) ** 2) - r0
    t = skfmm.travel_time(phi, v_grid, dx=grid.dx, order=2)
    # Corrección del radio inicial: tiempo desde la fuente hasta el círculo de radio r0
    ix = int(round((src_xz[0] - grid.xmin) / grid.dx))
    iz = int(round((src_xz[1] - grid.zmin) / grid.dx))
    t = np.asarray(t) + r0 / v_grid[iz, ix]
    inside = phi < 0
    t[inside] = np.sqrt((X[inside] - src_xz[0]) ** 2 + (Z[inside] - src_xz[1]) ** 2) / v_grid[iz, ix]
    return t


def traveltimes(v_grid, grid: Grid, src, rec):
    """Matriz de tiempos T[i_src, i_rec] y campos completos T[i_src, iz, ix]."""
    fields, tt = [], np.zeros((len(src), len(rec)))
    for i, s in enumerate(src):
        t = eikonal_fmm(v_grid, grid, s)
        interp = RegularGridInterpolator((grid.z, grid.x), t)
        tt[i] = interp(rec[:, ::-1])
        fields.append(t)
    return tt, np.stack(fields)
