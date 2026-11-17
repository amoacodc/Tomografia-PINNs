"""Modelo sintético 3D y problema directo (eikonal 3D con Fast Marching).

Convención: coordenadas (x, y, z) en km, z positiva hacia abajo.
Los campos en malla tienen forma (nz, ny, nx).
"""
from dataclasses import dataclass

import numpy as np
import skfmm
from scipy.interpolate import RegularGridInterpolator


@dataclass
class Grid3D:
    L: float = 1.0   # lado del cubo (km)
    n: int = 101

    @property
    def axis(self):
        return np.linspace(0.0, self.L, self.n)

    @property
    def dx(self):
        return self.L / (self.n - 1)

    def mesh(self):
        a = self.axis
        Z, Y, X = np.meshgrid(a, a, a, indexing="ij")
        return X, Y, Z


def true_velocity(x, y, z):
    """Gradiente vertical + anomalía lenta + anomalía rápida (km/s)."""
    v = 2.0 + 1.0 * z
    v = v - 0.6 * np.exp(-((x - 0.55) ** 2 + (y - 0.50) ** 2 + (z - 0.45) ** 2) / (2 * 0.12 ** 2))
    v = v + 0.5 * np.exp(-((x - 0.35) ** 2 + (y - 0.35) ** 2 + (z - 0.75) ** 2) / (2 * 0.10 ** 2))
    return v


ANOMALIES = {"lenta": ((0.55, 0.50, 0.45), -0.6, 0.12), "rapida": ((0.35, 0.35, 0.75), 0.5, 0.10)}


def anomaly_metrics(v_inv, grid):
    """Para cada anomalía: amplitud recuperada en su centro (respecto al fondo 2 + z)
    como % de la verdadera, y RMSE dentro de una esfera de radio 2 sigma."""
    X, Y, Z = grid.mesh()
    v_true = true_velocity(X, Y, Z)
    out = {}
    for name, (c, amp, sig) in ANOMALIES.items():
        ix, iy, iz = (int(round(ci / grid.dx)) for ci in c)
        rec_amp = v_inv[iz, iy, ix] - (2.0 + 1.0 * c[2])
        mask = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 < (2 * sig) ** 2
        out[f"anomalia_{name}_amplitud_recuperada_%"] = float(100 * rec_amp / amp)
        out[f"anomalia_{name}_rmse_km_s"] = float(np.sqrt(np.mean((v_inv[mask] - v_true[mask]) ** 2)))
    return out


def _well(x, y, z_vals):
    return np.stack([np.full_like(z_vals, x), np.full_like(z_vals, y), z_vals], axis=1)


def geometry_3d(n_src_well=6, n_rec_well=16, n_surf=8):
    """Fuentes en 2 pozos (x≈0); receptores en 3 pozos (x≈1 y y≈1) y malla en superficie."""
    zs = np.linspace(0.1, 0.9, n_src_well)
    src = np.concatenate([_well(0.02, 0.25, zs), _well(0.02, 0.75, zs)])
    zr = np.linspace(0.03, 0.97, n_rec_well)
    rec_w = np.concatenate([_well(0.98, 0.25, zr), _well(0.98, 0.75, zr), _well(0.50, 0.98, zr)])
    g = np.linspace(0.1, 0.9, n_surf)
    gx, gy = np.meshgrid(g, g)
    rec_s = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    return src, np.concatenate([rec_w, rec_s])


def eikonal_fmm(v_grid, grid: Grid3D, s):
    X, Y, Z = grid.mesh()
    r0 = 2.0 * grid.dx
    r = np.sqrt((X - s[0]) ** 2 + (Y - s[1]) ** 2 + (Z - s[2]) ** 2)
    phi = r - r0
    ix, iy, iz = (int(round(c / grid.dx)) for c in s)
    v_s = v_grid[iz, iy, ix]
    t = np.asarray(skfmm.travel_time(phi, v_grid, dx=grid.dx, order=2)) + r0 / v_s
    inside = phi < 0
    t[inside] = r[inside] / v_s
    return t


def traveltimes(v_grid, grid: Grid3D, src, rec):
    a = grid.axis
    tt = np.zeros((len(src), len(rec)))
    for i, s in enumerate(src):
        t = eikonal_fmm(v_grid, grid, s)
        tt[i] = RegularGridInterpolator((a, a, a), t)(rec[:, ::-1])  # (z, y, x)
    return tt
