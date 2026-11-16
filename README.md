# Tomografía sísmica de tiempos de viaje con PINN

Inversión de un modelo de velocidad 2D a partir de tiempos de primera llegada usando una
**red neuronal informada por la física (PINN)** que resuelve la ecuación eikonal
(enfoque tipo *PINNtomo*, Waheed et al., 2021).

## Física

Ecuación eikonal: `|∇T(x; xs)| = 1 / v(x)`.

Se usa la forma **factorizada** `T = T0 · τ`, con `T0 = |x − xs|`. Así la condición de
fuente `T(xs) = 0` se cumple por construcción y la singularidad del frente de onda en la
fuente queda absorbida en `T0`.

Dos redes se entrenan **a la vez**:

| Red | Entrada | Salida |
|-----|---------|--------|
| `tau_net` | `(x, z, xs, zs)` | `τ > 0` → tiempo de viaje para cualquier fuente |
| `v_net`   | `(x, z)`         | `v ∈ [vmin, vmax]` → **modelo de velocidad invertido** |

Pérdida:

```
L = w_pde · ‖ v·|∇T| − 1 ‖²        (residuo eikonal en puntos de colocación aleatorios)
  + w_data · ‖ T(xr; xs) − T_obs ‖²  (tiempos observados en receptores)
  + w_reg · ‖ ∇v ‖²                 (suavizado opcional)
```

No hace falta trazado de rayos ni cálculo de la matriz de sensibilidad: el gradiente
respecto a `v` sale de la diferenciación automática.

## Experimento sintético

* Dominio 1 km × 1 km; velocidad = gradiente vertical (2 → 3 km/s) + anomalía lenta
  (−0.6 km/s) + anomalía rápida (+0.5 km/s).
* Geometría entre pozos: 12 fuentes en el pozo izquierdo, 31 receptores en el derecho y 21
  en superficie → 624 tiempos, con ruido gaussiano de 1 ms.
* Los datos "observados" se generan con Fast Marching de 2º orden (`scikit-fmm`), que
  también se usa para **validar de forma independiente** el modelo invertido.
* Modelo inicial: velocidad constante 2.5 km/s.

### Resultados (configuración por defecto, ~2.5 min en CPU)

| Métrica | Valor |
|---|---|
| RMS tiempos, modelo inicial | 37.8 ms |
| RMS tiempos, modelo PINN (recalculado con FMM) | **1.58 ms** (ruido = 1 ms) |
| Error relativo de velocidad | 2.9 % |

![velocidad](outputs/velocidad.png)
![entrenamiento](outputs/entrenamiento.png)

La anomalía lenta y el gradiente se recuperan bien; la anomalía rápida profunda queda
suavizada, como es típico por la cobertura limitada de rayos en geometría entre pozos.

## Uso

```bash
pip install -r requirements.txt
python run_tomography.py                       # configuración por defecto
python run_tomography.py --iters 15000         # más entrenamiento
python run_tomography.py --v-fourier 32        # características de Fourier en v_net
python run_tomography.py --noise 0.003 --w-reg 1e-3
```

Salidas en `outputs/`: `velocidad.png`, `entrenamiento.png`, `ajuste_tiempos.png`,
`metrics.json`, `results.npz` y los pesos `model.pt`.

## Estructura

```
pinn_tomo/forward.py   # modelo verdadero, geometría y solver eikonal FMM (datos + validación)
pinn_tomo/pinn.py      # redes tau_net / v_net, residuo eikonal, regularización
run_tomography.py      # generación de datos, entrenamiento, evaluación y figuras
```

Para usar datos reales: sustituir `src`, `rec` y `t_obs` en `run_tomography.py` (normalizando
coordenadas al cuadrado [0, 1]²) y ajustar `--vmin/--vmax`.
