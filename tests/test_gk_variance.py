import numpy as np
import pandas as pd
from src.ingest_yahoo_gk import gk_variance

def test_gk_scale_invariance():
    ohlc = pd.DataFrame({
        "Open":[100,102,101,103],
        "High":[101,103,102,104],
        "Low":[ 99,101,100,102],
        "Close":[100,102,101,103],
    })
    v1 = gk_variance(ohlc).to_numpy()
    v2 = gk_variance(ohlc * 10.0).to_numpy()  # scale prices ×10
    assert np.allclose(v1, v2, rtol=0, atol=0)
