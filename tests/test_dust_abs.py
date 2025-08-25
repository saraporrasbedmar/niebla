import numpy as np
from numpy.testing import assert_allclose
import pytest

import src.niebla.dust_absorption_models as dust


def test_dust_att_finke():
    wv = np.array([0., 5., 10.])
    zz = np.array([0., 5., 10.])
    yy_calc = dust.dust_att_finke(wv, verbose=True)
    assert_allclose(yy_calc,
                    [np.nan, 0.05317699, 0.12612067],
                    atol=1e-6)

# TODO: write more tests here :D
# def test_sfr_constant():
#     zz = np.array([0., 5., 10.])
#     yy_calc = metall._metall_constant(zz, verbose=True)
#     assert_allclose(yy_calc,
#                     [0.02, 0.02, 0.02],
#                     atol=1e-6)
#
#
# def test_sfr_custom():
#     zz = np.array([0., 5., 10.])
#
#     with pytest.raises(ValueError):
#         metall._metall_custom(zz, metall_model=0., verbose=True)
#
#     with pytest.raises(NameError):
#         metall._metall_custom(zz, metall_model='z', verbose=True)
