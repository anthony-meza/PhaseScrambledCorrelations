import numpy as np
import pytest

from phase_scrambled_corr import AR1_simulate

def test_AR1_simulate_length_and_reproducibility():
    N = 250               # number of time points
    dt = 1.0              # time step = 1 month
    fs = 1 / dt           # sampling frequency = 1 / 1 month

    rho = 0.9
    noise_std = 1.0
    y0 = 1

    # run twice with the same seed
    t1, x1 = AR1_simulate(rho, noise_std, y0, N, seed=42)
    t2, x2 = AR1_simulate(rho, noise_std, y0, N, seed=42)

    # 1) length checks
    assert len(t1) == N
    assert len(x1) == N

    # 2) time array starts at 0 and increments by dt
    assert np.allclose(t1, np.arange(N) * dt)

    # 3) reproducibility: same seed ⇒ identical output
    assert np.array_equal(x1, x2)
