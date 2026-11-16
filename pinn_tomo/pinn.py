"""PINN para tomografía (2D o 3D: `dim` = nº de coordenadas espaciales) de tiempos de viaje (inspirado en PINNtomo, Waheed et al. 2021).

Dos redes:
  * tau_net(x, z, xs, zs) -> tau > 0, con T = T0 * tau y T0 = |x - xs| (eikonal factorizada).
    La condición de fuente T(xs) = 0 se cumple por construcción y la singularidad
    del frente de onda en la fuente queda absorbida en T0.
  * v_net(x, z) -> v en [vmin, vmax]: el modelo de velocidad que se invierte.

Pérdida = w_pde * || v |grad T| - 1 ||^2   (residuo eikonal en puntos de colocación)
        + w_data * || T(xr, xs) - T_obs ||^2   (tiempos observados en los receptores)
        + w_reg * || grad v ||^2               (suavizado opcional)
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, width, depth, act=nn.Tanh):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class VelocityNet(nn.Module):
    """v(x, z) acotada en [vmin, vmax].

    Con ``n_fourier > 0`` la entrada se proyecta a características de Fourier
    aleatorias [sin(2*pi*B x), cos(2*pi*B x)], B ~ N(0, sigma^2), lo que reduce el
    sesgo espectral del MLP y permite recuperar anomalías más compactas.
    """

    def __init__(self, vmin, vmax, width=32, depth=4, n_fourier=0, sigma=2.0, dim=2):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax
        self.n_fourier = n_fourier
        if n_fourier > 0:
            self.register_buffer("B", sigma * torch.randn(dim, n_fourier))
        n_in = 2 * n_fourier if n_fourier > 0 else dim
        self.mlp = MLP(n_in, 1, width, depth)
        nn.init.zeros_(self.mlp.net[-1].weight)  # arranca con v constante = (vmin+vmax)/2

    def features(self, xz):
        if self.n_fourier == 0:
            return 2 * xz - 1
        proj = 2 * torch.pi * xz @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)

    def forward(self, xz):
        return self.vmin + (self.vmax - self.vmin) * torch.sigmoid(self.mlp(self.features(xz)))


class TauNet(nn.Module):
    def __init__(self, s_ref, width=64, depth=6, dim=2):
        super().__init__()
        self.s_ref = s_ref
        self.mlp = MLP(2 * dim, 1, width, depth)

    def forward(self, xz, sxz):
        return self.s_ref * torch.exp(self.mlp(torch.cat([2 * xz - 1, 2 * sxz - 1], dim=1)))


def traveltime(tau_net, xz, sxz, eps=1e-8):
    t0 = torch.sqrt(((xz - sxz) ** 2).sum(dim=1, keepdim=True) + eps)
    return t0 * tau_net(xz, sxz)


def eikonal_residual(tau_net, v_net, xz, sxz):
    xz = xz.requires_grad_(True)
    t = traveltime(tau_net, xz, sxz)
    grad_t = torch.autograd.grad(t.sum(), xz, create_graph=True)[0]
    v = v_net(xz)
    return v * torch.sqrt((grad_t ** 2).sum(dim=1, keepdim=True) + 1e-12) - 1.0


def velocity_roughness(v_net, xz):
    xz = xz.requires_grad_(True)
    v = v_net(xz)
    g = torch.autograd.grad(v.sum(), xz, create_graph=True)[0]
    return (g ** 2).sum(dim=1).mean()
