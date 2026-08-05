import numpy as np
import pytest

from padic_kernel.profiles import (
    Profile,
    delta_profile,
    geometric_profile,
    linear_profile,
    uniform_profile,
)


def test_geometric_profile_values():
    f = geometric_profile(n=3, p=2, s=1.0)
    assert f.values == pytest.approx((0.125, 0.25, 0.5, 1.0))
    assert f.is_strictly_monotone


def test_amplitudes_normalise_and_reproduce_profile():
    for f in (geometric_profile(4, 3, 1.5), linear_profile(5)):
        a = f.amplitudes()
        assert np.sum(a**2) == pytest.approx(1.0, abs=1e-12)
        for v in range(f.height + 1):
            # overlap of two leaves whose LCA depth is v
            assert np.sum(a[: v + 1] ** 2) ** 2 == pytest.approx(f(v), abs=1e-12)


def test_non_monotone_profiles_are_flagged():
    assert not uniform_profile(3).is_strictly_monotone
    assert not delta_profile(3).is_strictly_monotone


def test_amplitudes_reject_non_monotone():
    with pytest.raises(ValueError, match="strictly monotone"):
        uniform_profile(3).amplitudes()


def test_profile_requires_terminal_one():
    with pytest.raises(ValueError, match=r"f\(n\) = 1"):
        Profile(values=(0.3, 0.7))


def test_call_is_vectorised():
    f = geometric_profile(3, 2, 1.0)
    out = f(np.array([0, 2, 3]))
    assert out == pytest.approx(np.array([0.125, 0.5, 1.0]))
