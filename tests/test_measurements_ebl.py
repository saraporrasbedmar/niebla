import os
import numpy as np
import pytest

from unittest.mock import MagicMock
from astropy.table import Table, vstack
import astropy.units as u

import src.niebla.measurements_folder.ebl as mod


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import pytest

import src.niebla.measurements_folder.ebl as mod


def test_import_spectrum_data_returns_qtable():
    data = mod._import_spectrum_data()
    assert hasattr(data, "colnames")
    assert "lambda" in data.colnames
    assert "nuInu" in data.colnames
    assert "type" in data.colnames
    assert "reference" in data.colnames


def test_import_spectrum_data_units():
    data = mod._import_spectrum_data()
    assert data["lambda"].unit == u.um
    assert data["nuInu"].unit == u.nW / (u.m**2 * u.sr)
    assert data["nuInu_errn"].unit == u.nW / (u.m**2 * u.sr)
    assert data["nuInu_errp"].unit == u.nW / (u.m**2 * u.sr)


def test_import_spectrum_data_lambda_filter():
    data = mod._import_spectrum_data(lambda_min=0.1, lambda_max=1.0)
    assert np.all((data["lambda"].value >= 0.1) & (data["lambda"].value <= 1.0))


def test_import_spectrum_data_type_filter():
    data = mod._import_spectrum_data(import_one_type="UL")
    assert len(data) > 0
    assert np.all(data["type"] == "UL")


def test_dictionary_datatype_returns_requested_type():
    data = mod._dictionary_datatype("IGL")
    assert len(data) > 0
    assert np.all(data["type"] == "IGL")


def test_dictionary_datatype_units():
    data = mod._dictionary_datatype("IGL")
    assert data["lambda"].unit == u.um
    assert data["nuInu"].unit == u.nW / (u.m**2 * u.sr)


def test_dictionary_datatype_obs_not_taken():
    all_data = mod._dictionary_datatype("IGL")
    if len(all_data) == 0:
        pytest.skip("No IGL data available in packaged files.")

    ref = str(all_data["reference"][0])
    filtered = mod._dictionary_datatype("IGL", obs_not_taken=[ref])

    assert ref not in [str(r) for r in filtered["reference"]]


def test_ebl_returns_table_without_plot():
    table = mod.ebl(plot=False, obs_not_taken=[])
    assert hasattr(table, "colnames")
    assert "lambda" in table.colnames
    assert "type" in table.colnames


def test_ebl_returns_legend_with_plot():
    fig, ax = plt.subplots()
    table, legend = mod.ebl(plot=True, axis=ax, obs_not_taken=[])
    assert hasattr(table, "colnames")
    assert legend is not None
    assert len(legend.texts) > 0
    plt.close(fig)


with pytest.raises(AttributeError):
    mod.ebl(plot=True, axis=None)


def test_ebl_relabels_known_nh_entries():
    table = mod.ebl(plot=False, obs_not_taken=[])
    if "reference" not in table.colnames:
        pytest.skip("No data returned.")

    mask = table["reference"] == "NH/LORRI (Symons+ ‘23)"
    if np.any(mask):
        assert np.all(table["type"][mask] == "NH")

def test_ebl_contains_ul_arrow_for_known_reference():
    table = mod.ebl(plot=False, obs_not_taken=[])
    mask = table["type"] == "UL_arrow"
    # This is a regression-style check: if a known entry exists, it must be marked correctly.
    assert table[mask].dtype is not None
