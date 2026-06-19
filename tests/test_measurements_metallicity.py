import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.niebla.measurements_folder.metallicity as mod


def test_metallicity_returns_table():
    table = mod.metallicity()

    assert len(table) == 7
    assert table.colnames == ["z", "metall", "metall_err_low", "metall_err_up"]


def test_metallicity_values_are_finite():
    table = mod.metallicity()

    for col in table.colnames:
        assert np.all(np.isfinite(table[col]))


def test_metallicity_uses_default_z_sun():
    table = mod.metallicity()

    # First row corresponding to z = 0.55
    assert table["metall"][0] > 0


def test_metallicity_custom_z_sun_changes_values():
    table_default = mod.metallicity(z_sun=0.02)
    table_custom = mod.metallicity(z_sun=0.01)

    assert not np.allclose(table_default["metall"], table_custom["metall"])


def test_metallicity_plotting_runs():
    fig, ax = plt.subplots()

    table = mod.metallicity(axis=ax)

    assert len(table) == 7
    assert len(ax.collections) > 0 or len(ax.lines) > 0

    plt.close(fig)


def test_metallicity_plotting_with_label():
    fig, ax = plt.subplots()

    table = mod.metallicity(axis=ax, label="My label")

    assert len(table) == 7
    plt.close(fig)