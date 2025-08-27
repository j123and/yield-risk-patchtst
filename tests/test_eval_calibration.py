import numpy as np
from src.eval_phase4 import _calibrate_fixed, _calibrate_rolling, _exception_indicator

def test_exception_indicator():
    ret = np.array([0.0, -0.1, 0.02])
    q   = np.array([-0.05, -0.08, 0.0])
    ex = _exception_indicator(ret, q)
    assert ex.tolist() == [0, 1, 0]

def test_calibrate_fixed_recovers_alpha_on_bias():
    rng = np.random.default_rng(0)
    n = 400; alpha = 0.05
    # true residuals ~ N(0,1), biased predictions q = true_quantile - 0.2
    eps = rng.normal(size=n)
    ret = eps
    q   = np.quantile(eps, alpha) - 0.2 + np.zeros(n)
    ex_fixed, rate, N = _calibrate_fixed(ret, q, alpha, window=200)
    # after calibration, rate ~ alpha on the post-warmup region
    assert N == n - 200
    assert abs(rate - alpha) < 0.02

def test_calibrate_rolling_tracks_drift():
    rng = np.random.default_rng(1)
    n = 500; alpha = 0.05
    eps = rng.normal(size=n)
    # drifting bias: q underestimates by a slowly varying delta
    drift = np.linspace(-0.3, 0.3, n)
    ret = eps
    q   = np.quantile(eps, alpha) + drift
    ex_roll, rate, N = _calibrate_rolling(ret, q, alpha, window=100, ema=0.5)
    assert N == n - 100
    assert abs(rate - alpha) < 0.03
