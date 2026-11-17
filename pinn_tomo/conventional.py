"""Tomografía convencional de tiempos de viaje con rayos curvos (2D y 3D).

Inversión linealizada iterativa (Gauss-Newton):
  1. Tiempos T(x; xs) con Fast Marching sobre el modelo actual.
  2. Rayos curvos: se trazan desde cada receptor bajando por -grad T hasta la fuente.
  3. Matriz de sensibilidad G (longitud de rayo repartida con pesos multilineales en los
     nodos de una malla gruesa de lentitud s = 1/v).
  4. Mínimos cuadrados regularizados (LSQR):
        min || G ds - r ||^2 + lam^2 || L (s + ds) ||^2 + eps^2 || ds ||^2
     con r = T_obs - T_calc y L el laplaciano discreto (suavizado).
  5. s <- s + ds, se repite.

Coordenadas de los puntos en orden (x, z) en 2D y (x, y, z) en 3D; los campos en malla
tienen los ejes al revés: (z, x) y (z, y, x).
"""
import time
from itertools import product

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import lsqr


class NodeGrid:
    """Parametrización de la lentitud en una malla gruesa de nodos en [0, 1]^dim."""

    def __init__(self, n, dim):
        self.n, self.dim = n, dim
        self.axis = np.linspace(0.0, 1.0, n)
        self.h = 1.0 / (n - 1)
        self.shape = (n,) * dim          # orden de ejes (z, [y,] x)
        self.size = n ** dim

    def weights(self, pts):
        """Pesos multilineales: devuelve (fila_local, índice_nodo, peso) para cada esquina."""
        f = np.clip(pts / self.h, 0, self.n - 1 - 1e-9)
        i0 = np.floor(f).astype(int)
        t = f - i0
        rows, cols, vals = [], [], []
        for corner in product((0, 1), repeat=self.dim):
            c = np.array(corner)
            idx = i0 + c
            w = np.prod(np.where(c == 1, t, 1 - t), axis=1)
            flat = np.ravel_multi_index(tuple(idx[:, ::-1].T), self.shape)
            rows.append(np.arange(len(pts))); cols.append(flat); vals.append(w)
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    def to_fine(self, s_nodes, fine_axis):
        interp = RegularGridInterpolator((self.axis,) * self.dim, s_nodes.reshape(self.shape))
        mesh = np.meshgrid(*([fine_axis] * self.dim), indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], axis=1)   # ya en orden (z, [y,] x)
        return interp(pts).reshape((len(fine_axis),) * self.dim)

    def laplacian(self):
        n = self.n
        d2 = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="lil")
        d2[0, :2] = [-1, 1]; d2[-1, -2:] = [1, -1]          # bordes tipo Neumann
        d2 = d2.tocsr()
        eye = sp.identity(n, format="csr")
        L = None
        for k in range(self.dim):
            mats = [d2 if j == k else eye for j in range(self.dim)]
            term = mats[0]
            for m in mats[1:]:
                term = sp.kron(term, m, format="csr")
            L = term if L is None else L + term
        return L


def trace_rays(T, fine_axis, src, rec, step):
    """Traza rayos desde cada receptor hasta la fuente siguiendo -grad T.

    Devuelve (id_rayo, punto_medio, longitud) de todos los segmentos.
    """
    dim = len(src)
    h = fine_axis[1] - fine_axis[0]
    grads = np.gradient(T, h)                             # derivadas en orden (z, [y,] x)
    interps = [RegularGridInterpolator((fine_axis,) * dim, g, bounds_error=False, fill_value=None)
               for g in grads]
    p = rec.astype(float).copy()
    alive = np.ones(len(rec), bool)
    ids, mids, lens = [], [], []
    max_steps = int(4.0 / step)
    for _ in range(max_steps):
        if not alive.any():
            break
        ia = np.where(alive)[0]
        q = p[ia]
        g = np.stack([I(q[:, ::-1]) for I in interps], axis=1)[:, ::-1]   # vuelve a (x, [y,] z)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        to_src = src - q
        dist = np.linalg.norm(to_src, axis=1)
        near = dist <= 1.5 * step
        new = np.clip(q - step * g, 0.0, 1.0)
        new[near] = src                                   # último tramo recto hasta la fuente
        seg = np.linalg.norm(new - q, axis=1)
        ids.append(ia); mids.append(0.5 * (q + new)); lens.append(seg)
        p[ia] = new
        alive[ia[near]] = False
    return np.concatenate(ids), np.concatenate(mids), np.concatenate(lens)


def invert(t_obs, src, rec, fmm, fine_axis, n_nodes=26, n_iter=8, lam=1.0, eps=0.05,
           v0=2.5, vmin=1.0, vmax=4.0, verbose=True):
    """Inversión convencional. ``fmm(v_fine, s)`` debe devolver el campo T para la fuente s.

    Devuelve (v en malla fina, historial de RMS en ms).
    """
    dim = src.shape[1]
    ng = NodeGrid(n_nodes, dim)
    L = ng.laplacian()
    s = np.full(ng.size, 1.0 / v0)
    step = 0.5 * (fine_axis[1] - fine_axis[0])
    nrec = len(rec)
    hist = []
    for it in range(n_iter + 1):
        t0 = time.time()
        v_f = 1.0 / ng.to_fine(s, fine_axis)
        t_calc = np.zeros((len(src), nrec))
        rows, cols, vals = [], [], []
        for i, s_xyz in enumerate(src):
            T = fmm(v_f, s_xyz)
            interp = RegularGridInterpolator((fine_axis,) * dim, T)
            t_calc[i] = interp(rec[:, ::-1])
            if it == n_iter:
                continue
            rid, mid, ln = trace_rays(T, fine_axis, s_xyz, rec, step)
            r_loc, c, w = ng.weights(mid)
            rows.append(i * nrec + rid[r_loc]); cols.append(c); vals.append(w * ln[r_loc])
        resid = (t_obs - t_calc).ravel()
        rms = 1e3 * np.sqrt(np.mean(resid ** 2))
        hist.append(rms)
        if verbose:
            print(f"  iter {it:2d} | RMS {rms:6.2f} ms | {time.time() - t0:5.1f} s", flush=True)
        if it == n_iter:
            break
        G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(src) * nrec, ng.size))
        A = sp.vstack([G, lam * L, eps * sp.identity(ng.size)], format="csr")
        b = np.concatenate([resid, -lam * (L @ s), np.zeros(ng.size)])
        ds = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
        s = np.clip(s + ds, 1.0 / vmax, 1.0 / vmin)
    return 1.0 / ng.to_fine(s, fine_axis), hist
