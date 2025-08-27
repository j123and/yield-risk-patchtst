import numpy as np
import pytest
from scipy.stats import chi2
from src.eval_phase4 import kupiec_pvalue, christoffersen_ind_pvalue, christoffersen_cc_pvalue

def _lr_uc(ex, alpha):
    n = len(ex); x = int(ex.sum())
    ll_alpha = (n - x) * np.log1p(-alpha) + x * np.log(alpha)
    pi = x / n if n else 0.0
    ll_pi = (n - x) * np.log1p(-pi + 1e-300) + x * np.log(pi + 1e-300)
    return -2.0 * (ll_alpha - ll_pi)

def _lr_ind(ex):
    e = ex.astype(int)
    n00=n01=n10=n11=0
    for i in range(1, len(e)):
        a,b = e[i-1], e[i]
        if   a==0 and b==0: n00+=1
        elif a==0 and b==1: n01+=1
        elif a==1 and b==0: n10+=1
        else:               n11+=1
    n0 = n00 + n01; n1 = n10 + n11
    if n0==0 or n1==0: return 0.0
    pi01 = n01 / n0; pi11 = n11 / n1
    pi1  = (n01 + n11) / (n0 + n1)
    ll_ind = 0.0
    if pi01>0 and (1-pi01)>0: ll_ind += n01*np.log(pi01) + n00*np.log1p(-pi01)
    if pi11>0 and (1-pi11)>0: ll_ind += n11*np.log(pi11) + n10*np.log1p(-pi11)
    ll_unr = ll_ind
    ll_r   = (n01+n11)*np.log(pi1 + 1e-300) + (n00+n10)*np.log1p(-pi1 + 1e-300)
    return -2.0 * (ll_r - ll_unr)

def test_kupiec_matches_chi2_sf():
    rng = np.random.default_rng(0)
    alpha = 0.05
    ex = (rng.random(1000) < alpha).astype(int)
    lr = _lr_uc(ex, alpha)
    p_true = chi2.sf(lr, df=1)
    p_func = kupiec_pvalue(ex, alpha)
    assert np.isclose(p_func, p_true, rtol=1e-6, atol=1e-12)

def test_christoffersen_independence_matches_chi2_sf():
    rng = np.random.default_rng(1)
    ex = (rng.random(1200) < 0.05).astype(int)
    lr = _lr_ind(ex)
    p_true = chi2.sf(lr, df=1)
    p_func = christoffersen_ind_pvalue(ex)
    assert np.isclose(p_func, p_true, rtol=1e-6, atol=1e-12)

def test_christoffersen_cc_matches_chi2_sf():
    rng = np.random.default_rng(2)
    ex = (rng.random(900) < 0.05).astype(int)
    # CC = UC + IND
    lr_uc = _lr_uc(ex, 0.05)
    lr_ind = _lr_ind(ex)
    p_true = chi2.sf(lr_uc + lr_ind, df=2)
    p_func = christoffersen_cc_pvalue(ex, 0.05)
    assert np.isclose(p_func, p_true, rtol=1e-6, atol=1e-12)
