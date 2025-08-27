import numpy as np
import pandas as pd
import pytest
from src.build_sequences import main as build_main, _choose_close_col, make_windows

def test_choose_close_prefers_adj_close():
    df = pd.DataFrame({"adj_close":[1,2], "Close":[1,2]})
    assert _choose_close_col(df, allow_raw=True) == "adj_close"

def test_choose_close_requires_flag_for_raw_close():
    df = pd.DataFrame({"Close":[1,2]})
    with pytest.raises(ValueError):
        _choose_close_col(df, allow_raw=False)

def test_make_windows_alignment():
    # simple monotone prices with sigma_gk and RV_GK present
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "adj_close": np.linspace(100, 107, 8),
        "RV_GK": np.linspace(0.01, 0.02, 8),
        "sigma_gk": np.sqrt(np.linspace(0.01, 0.02, 8)),
    })
    X, y_ret, y_lrv, _ = make_windows(df, seq_len=3, close_col="adj_close")
    # N = len(df)-1 - seq_len
    assert X.shape == (4, 3, 2)  # features: ret, sigma_gk
    # target is next-day log return and next-day log RV
    assert y_ret.shape == (4,)
    assert y_lrv.shape == (4,)
