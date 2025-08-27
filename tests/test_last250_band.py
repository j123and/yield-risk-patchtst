import numpy as np
import pytest
from scipy.stats import binom
from src.eval_phase4 import last250_band

def test_last250_band_exact_binomial():
    lo, hi = last250_band(alpha=0.05, n=250, conf=0.95)
    lo_ref, hi_ref = binom.interval(0.95, 250, 0.05)
    assert (lo, hi) == (int(lo_ref), int(hi_ref))
