"""AR(1)-GBM alpha-injection tests.

Checks that the ARGBM model:
- recovers exact GBM behavior (iid log-returns) at phi=0,
- injects ~phi lag-1 autocorrelation at phi != 0,
- preserves per-step return variance for every phi (apples-to-apples),
- exposes a working Model interface (log_likelihood, fit, diagnostics),
- flows through ``data.generate(ar_coef=...)`` into the DataBundle.
"""

import mlx.core as mx
import numpy as np
import pytest

from dirty_mkt_data.core.argbm import ARGBM
from data import SYMBOLS, generate

TRADING_DAYS = 252


def _sample_returns(model, n_steps=20000, n_paths=1, seed=7):
    ds = model.sample(n_steps, n_paths=n_paths, key=mx.random.key(seed))
    return np.asarray(ds.returns)


def _autocorr1(returns: np.ndarray, paths_axis_last=False) -> np.ndarray:
    r = returns
    z = (r - r.mean(axis=-1, keepdims=True)) / (
        r.std(axis=-1, keepdims=True) + 1e-12
    )
    return np.mean(z[..., :-1] * z[..., 1:], axis=-1)


def test_phi_zero_matches_gbm():
    """phi=0 must have zero lag-1 autocorr and match GBM scale."""
    m = ARGBM(mu=0.3, sigma=0.6, s0=100.0, dt=1.0 / TRADING_DAYS, phi=0.0)
    r = _sample_returns(m)
    rho1 = float(np.asarray(_autocorr1(r))[0])
    scale = 0.6 / TRADING_DAYS**0.5
    assert 0.99 < r.std() / scale < 1.01
    assert abs(rho1) < 0.02


@pytest.mark.parametrize("phi", [0.3, -0.3, 0.7])
def test_phi_injects_autocorr(phi):
    m = ARGBM(mu=0.0, sigma=0.6, s0=100.0, phi=phi)
    r = _sample_returns(m, n_steps=50000)
    rho1 = float(np.asarray(_autocorr1(r))[0])
    # AR(1) sample autocorr is biased toward zero; accept a loose band.
    assert abs(rho1 - phi) < 0.06
    assert np.sign(rho1) == np.sign(phi)
    # Coarse theoretical std of the autocorr estimate under H1.
    assert 0.9 * abs(phi) - 0.05 <= abs(rho1)


def test_variance_preserved_across_phi():
    """All phi share the same per-step return std as GBM."""
    dt = 1.0 / TRADING_DAYS
    sigmas = []
    for phi in (0.0, 0.3, 0.6, -0.4):
        r = _sample_returns(ARGBM(mu=0.1, sigma=0.5, s0=100.0, dt=dt, phi=phi),
                            n_steps=30000)
        sigmas.append(r.std())
    target = 0.5 * dt**0.5
    assert max(abs(s - target) for s in sigmas) < 0.02 * target


def test_log_likelihood_peak_at_true_phi():
    """Conditional AR(1) likelihood should favor the injected phi."""
    dt = 1.0 / TRADING_DAYS
    r = _sample_returns(ARGBM(mu=0.1, sigma=0.5, s0=100.0, dt=dt, phi=0.5),
                        n_steps=20000)
    r = mx.array(r)
    lls = []
    for phi in (0.0, 0.3, 0.5, 0.7):
        m = ARGBM(mu=0.1, sigma=0.5, s0=100.0, dt=dt, phi=phi)
        lls.append(float(mx.sum(m.log_likelihood(r))))
    assert lls[2] > lls[0] and lls[2] > lls[1] and lls[2] > lls[3]


def test_fit_roundtrip_recovers_coefs():
    dt = 1.0 / TRADING_DAYS
    m = ARGBM(mu=0.2, sigma=0.4, s0=100.0, dt=dt, phi=0.4)
    r = mx.array(_sample_returns(m, n_steps=20000))
    fitted = m.fit(r)
    assert abs(fitted.phi - 0.4) < 0.03
    assert abs(fitted.mu - 0.2) < 0.02
    assert abs(fitted.sigma - 0.4) < 0.01


def test_diagnostics_exposes_ar1():
    m = ARGBM(mu=0.0, sigma=0.5, s0=100.0, phi=0.5)
    ds = m.sample(20000, key=mx.random.key(3))
    diag = m.diagnostics(ds)
    assert "ar1" in diag
    assert abs(float(diag["ar1"][0]) - 0.5) < 0.06


def test_generate_ar_coef_flows_into_bundle():
    """ar_coef > 0 changes the data: lag-1 autocorr should be positive."""
    b = generate(symbols={"BTC": SYMBOLS["BTC"]}, n_steps=2500, seed=11,
                 low_tf=5, high_tf=240, ar_coef=0.5)
    r = np.log(np.asarray(b.ohlcv.closes))
    r = np.diff(r, axis=1)
    rho1 = float(np.mean(_autocorr1(r)))
    assert rho1 > 0.10