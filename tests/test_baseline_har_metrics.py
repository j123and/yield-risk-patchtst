import numpy as np
from src.baseline_har import rmse, qlike

def test_rmse_simple():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.0, 2.0, 4.0])
    # errors = [0,0,1] -> MSE=1/3 -> RMSE = sqrt(1/3)
    assert np.isclose(rmse(y, yhat), (1/3)**0.5)

def test_qlike_properties():
    y = np.array([0.5, 1.0, 2.0])
    s2 = np.array([0.6, 0.9, 2.2])
    q = qlike(y, s2)
    assert np.isfinite(q) and q > 0.0
