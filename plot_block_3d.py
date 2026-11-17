"""Bloque 3D (block diagram) del modelo de velocidad: verdadero, PINN y error.

Cubo con un corte en esquina que deja ver el interior: la cara y = yc atraviesa la
anomalía lenta y el piso del corte z = zc muestra la anomalía rápida.
Genera un HTML interactivo (Plotly: rotar, zoom, valores al pasar el ratón).

Requiere outputs_3d/results.npz (se genera con run_tomography_3d.py).
    python plot_block_3d.py
    python plot_block_3d.py --cut 0.3 0.5 0.75
    python plot_block_3d.py --local-js --out docs/bloque_3d.html   # sitio de GitHub Pages
"""
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator


def block_faces(interp, cut, n=61):
    """Lista de (x, y, z, valores) para las caras visibles del cubo con la esquina cortada."""
    xc, yc, zc = cut
    lin = lambda a, b: np.linspace(a, b, max(2, int(round(n * (b - a))) + 1))

    def face(axis, const, u_rng, v_rng):
        u, v = np.meshgrid(lin(*u_rng), lin(*v_rng))
        c = np.full_like(u, const)
        x, y, z = {"x": (c, u, v), "y": (u, c, v), "z": (u, v, c)}[axis]
        return x, y, z, interp(np.stack([z, y, x], -1))

    return [
        # caras exteriores sin la esquina (x > xc, y < yc, z < zc)
        face("z", 0.0, (0, xc), (0, 1)), face("z", 0.0, (xc, 1), (yc, 1)),
        face("y", 0.0, (0, xc), (0, 1)), face("y", 0.0, (xc, 1), (zc, 1)),
        face("x", 1.0, (yc, 1), (0, 1)), face("x", 1.0, (0, yc), (zc, 1)),
        face("x", 0.0, (0, 1), (0, 1)), face("y", 1.0, (0, 1), (0, 1)), face("z", 1.0, (0, 1), (0, 1)),
        # caras interiores del corte
        face("z", zc, (xc, 1), (0, yc)), face("x", xc, (0, yc), (0, zc)), face("y", yc, (xc, 1), (0, zc)),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="outputs_3d/results.npz")
    p.add_argument("--cut", type=float, nargs=3, default=[0.30, 0.50, 0.75], metavar=("XC", "YC", "ZC"))
    p.add_argument("--out", default="outputs_3d/bloque_3d.html")
    p.add_argument("--inline-js", action="store_true", help="incrustar plotly.js (funciona sin internet)")
    p.add_argument("--local-js", action="store_true",
                   help="usar plotly.min.js en la misma carpeta que el HTML (para GitHub Pages)")
    a = p.parse_args()

    r = np.load(a.results)
    g = r["axis_km"]
    v_true, v_inv = r["v_true"], r["v_inv"]
    vmin, vmax = float(v_true.min()), float(v_true.max())
    panels = [("Modelo verdadero", v_true, "Jet_r", vmin, vmax),
              ("Modelo invertido (PINN)", v_inv, "Jet_r", vmin, vmax),
              ("Error (PINN − verdadero)", v_inv - v_true, "RdBu_r", -0.4, 0.4)]

    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        subplot_titles=[t for t, *_ in panels], horizontal_spacing=0.02)
    for k, (title, v, cs, lo, hi) in enumerate(panels):
        interp = RegularGridInterpolator((g, g, g), v)
        for j, (x, y, z, c) in enumerate(block_faces(interp, a.cut)):
            fig.add_trace(go.Surface(
                x=x, y=y, z=z, surfacecolor=c, colorscale=cs, cmin=lo, cmax=hi,
                showscale=(j == 0 and k > 0),
                colorbar=dict(title="km/s", len=0.6, x=0.655 if k == 1 else 1.0, thickness=14),
                lighting=dict(ambient=1, diffuse=0, specular=0),
                hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f} km<br>%{surfacecolor:.2f} km/s<extra></extra>",
            ), row=1, col=k + 1)
        if k < 2:
            for pts, sym, col, name in [(r["src"], "diamond", "red", "fuentes"),
                                        (r["rec"], "circle", "black", "receptores")]:
                fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                                           marker=dict(size=3 if col == "black" else 5, symbol=sym, color=col),
                                           name=name, showlegend=(k == 0)), row=1, col=k + 1)
    scene = dict(xaxis=dict(title="x (km)", range=[0, 1]), yaxis=dict(title="y (km)", range=[0, 1]),
                 zaxis=dict(title="z (km)", range=[1, 0]), aspectmode="cube",
                 camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1)))
    fig.update_layout(scene=scene, scene2=scene, scene3=scene, height=620, autosize=True,
                      title=f"Tomografía PINN 3D — bloque con corte en x={a.cut[0]}, y={a.cut[1]}, "
                            f"z={a.cut[2]} km (arrastra para rotar)",
                      legend=dict(x=0.01, y=0.02), margin=dict(l=10, r=10, t=70, b=10))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(a.out, include_plotlyjs=True if a.inline_js else ("directory" if a.local_js else "cdn"), config={"responsive": True})
    print(f"Guardado {a.out}")


if __name__ == "__main__":
    main()
