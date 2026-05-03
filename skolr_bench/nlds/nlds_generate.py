"""Generate the 4 nonlinear dynamical systems from SKOLR Appendix E.

Each: 20,000 timesteps, single trajectory.
Splits: 14000 train / 2000 val / 4000 test (per paper).

Saves npz files to skolr_bench/nlds_data/.
"""
from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'nlds_data')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOTAL = 20000
N_TRAIN, N_VAL, N_TEST = 14000, 2000, 4000
assert N_TRAIN + N_VAL + N_TEST == N_TOTAL


def pendulum(seed=0, dt=0.001, g=9.81, l=1.0):
    """θ'' + (g/l)·sin(θ) = 0; semi-implicit (symplectic) Euler so
    energy is approximately conserved. State: [theta, omega]."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi)
    omega = rng.uniform(-1.0, 1.0)
    out = np.empty((N_TOTAL, 2), dtype=np.float64)
    for k in range(N_TOTAL):
        out[k] = [theta, omega]
        omega = omega - (g / l) * np.sin(theta) * dt
        theta = theta + omega * dt
    return out


def duffing(seed=0, dt=0.001, delta=0.3, alpha=1.0, beta=5.0,
            gamma=8.0, omega=0.5):
    """ẍ + δẋ + αx + βx³ = γ·cos(ωt). Semi-implicit Euler. State: [x, xdot]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.1, 0.1)
    xdot = rng.uniform(-0.1, 0.1)
    out = np.empty((N_TOTAL, 2), dtype=np.float64)
    for k in range(N_TOTAL):
        out[k] = [x, xdot]
        t = k * dt
        xddot = -delta * xdot - alpha * x - beta * x ** 3 + gamma * np.cos(omega * t)
        xdot = xdot + xddot * dt
        x = x + xdot * dt
    return out


def lotka_volterra(seed=0, dt=0.001, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
    """dN_p/dt = αN_p - βN_p N_d ; dN_d/dt = δN_p N_d - γN_d. State: [N_p, N_d]."""
    rng = np.random.default_rng(seed)
    Np = rng.uniform(1.0, 5.0)
    Nd = rng.uniform(1.0, 5.0)
    out = np.empty((N_TOTAL, 2), dtype=np.float64)
    for k in range(N_TOTAL):
        out[k] = [Np, Nd]
        dNp = alpha * Np - beta * Np * Nd
        dNd = delta * Np * Nd - gamma * Nd
        Np = Np + dNp * dt
        Nd = Nd + dNd * dt
    return out


def lorenz63(seed=0, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3):
    """Classic Lorenz '63. State: [x, y, z]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0)
    y = rng.uniform(-1.0, 1.0)
    z = rng.uniform(-1.0, 1.0) + 25.0
    out = np.empty((N_TOTAL, 3), dtype=np.float64)
    for k in range(N_TOTAL):
        out[k] = [x, y, z]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x = x + dx * dt
        y = y + dy * dt
        z = z + dz * dt
    return out


SYSTEMS = {
    'pendulum':       (pendulum, ['theta', 'omega']),
    'duffing':        (duffing,  ['x', 'xdot']),
    'lotka_volterra': (lotka_volterra, ['N_prey', 'N_predator']),
    'lorenz63':       (lorenz63, ['x', 'y', 'z']),
}


def generate_and_save(seeds=(0, 1, 2, 3, 4)):
    for name, (fn, dims) in SYSTEMS.items():
        for seed in seeds:
            traj = fn(seed=seed)
            train = traj[:N_TRAIN]
            val   = traj[N_TRAIN:N_TRAIN + N_VAL]
            test  = traj[N_TRAIN + N_VAL:]
            out = os.path.join(OUT_DIR, f'{name}_seed{seed}.npz')
            np.savez(out, train=train, val=val, test=test, dims=dims)
        print(f"{name:>16s}  ({len(seeds)} seeds)  shape={traj.shape}  "
              f"last-seed range=[{traj.min():.3f}, {traj.max():.3f}]  "
              f"std={traj.std():.3f}")


if __name__ == "__main__":
    generate_and_save()
