import numpy as np
from PhaseScrambledCorrelations.ar_processes import AR1_process
import pytest


def test_AR1_simulate_length_and_reproducibility():
    N = 250               # number of time points
    dt = 1.0              # time step = 1 month

    rho = 0.9
    noise_std = 1.0
    y0 = 1

    # run twice with the same seed
    ts1 = AR1_process(rho, noise_std, y0, N, seed=42, dt = dt)
    ts2 = AR1_process(rho, noise_std, y0, N, seed=42, dt = dt)

    # 1) length checks
    assert ts1.n == N
    assert ts2.n == N

    # 2) time array starts at 0 and increments by dt
    assert np.allclose(ts1.time, np.arange(N) * dt)

    # 3) reproducibility: same seed ⇒ identical output
    assert np.array_equal(ts1.data, ts2.data)
