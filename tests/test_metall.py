import numpy as np
from numpy.testing import assert_allclose
import pytest

import src.niebla.metall_models as metall


def test_metall_tanikawa22():
    zz = np.array([0., 5., 10.])
    yy_calc = metall._metall_tanikawa22(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.02844658, 0.006524, 0.00068405],
                    atol=1e-6)


def test_sfr_constant():
    zz = np.array([0., 5., 10.])
    yy_calc = metall._metall_constant(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.02, 0.02, 0.02],
                    atol=1e-6)


def test_sfr_custom():
    zz = np.array([0., 5., 10.])

    with pytest.raises(ValueError):
        metall._metall_custom(zz, metall_model=0., verbose=True)

    with pytest.raises(NameError):
        metall._metall_custom(zz, metall_model='z', verbose=True)
