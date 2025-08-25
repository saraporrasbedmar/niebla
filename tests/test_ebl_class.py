import numpy as np
from numpy.testing import assert_allclose
import pytest

import src.niebla.ebl_model as ebl


def test_sfr_madau14():
    wv = np.array([0., 5., 10.])
    zz = np.array([0., 5., 10.])
    inputs_dict = {
        'cosmology_params':
            {'cosmo': [ .7, .3, .7 ], 'omegaBar': 0.0453},
        'wavelength_array':
            {'wv_min': 1e-2, 'wv_max': 1e5, 'wv_steps': 700},
        'redshift_array':
            {'z_min': 0., 'z_max': 10., 'z_steps': 100},
        'z_intmax': 40.,
        't_intsteps': 201,
        'log_prints': True}

    y_calc = ebl.EBL_model.input_yaml_data_into_class(
        yaml_data=inputs_dict)

# TODO: write more tests here :D

# def test_sfr_custom():
#     zz = np.array([0., 5., 10.])
#
#     with pytest.raises(ValueError):
#         sfr._sfr_custom(zz, sfr_model=0., verbose=True)
#
#     with pytest.raises(NameError):
#         sfr._sfr_custom(zz, sfr_model='z', verbose=True)
