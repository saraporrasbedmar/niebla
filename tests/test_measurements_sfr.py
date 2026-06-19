import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.niebla.measurements as mod


def test_sfr_returns_astropy_table_without_plot():
    table = mod.sfr(plot=False)

    assert len(table) > 0
    assert "z_value" in table.colnames
    assert "zerr_low" in table.colnames
    assert "zerr_up" in table.colnames
    assert "sfr_value" in table.colnames
    assert "sfrerr_low" in table.colnames
    assert "sfrerr_up" in table.colnames
    assert "Reference" in table.colnames


def test_sfr_table_has_expected_columns_types():
    table = mod.sfr(plot=False)

    numeric_cols = [
        "z_value", "zerr_low", "zerr_up",
        "sfr_value", "sfrerr_low", "sfrerr_up"
    ]

    for col in numeric_cols:
        assert np.all(np.isfinite(table[col]))


def test_sfr_contains_expected_references():
    table = mod.sfr(plot=False)

    refs = set(table["Reference"])
    expected = {
        "Madau $&$ Dickinson UV data",
        "Madau $&$ Dickinson IR data",
        "Driver et al. 2018",
        "Bourne et al. 2017",
        "Bouwens et al. 2015",
    }

    assert expected.issubset(refs)


def test_sfr_row_counts_are_reasonable():
    table = mod.sfr(plot=False)

    # Exact counts are useful if the data is stable.
    # Replace these with the true expected counts if you want stricter tests.
    assert len(table) >= 5


def test_sfr_plotting_does_not_crash():
    fig, ax = plt.subplots()

    table = mod.sfr(plot=True, axis=ax)

    assert len(table) > 0
    assert len(ax.lines) > 0 or len(ax.collections) > 0

    plt.close(fig)


def test_sfr_plotting_with_custom_markers_and_colors():
    fig, ax = plt.subplots()

    table = mod.sfr(
        plot=True,
        axis=ax,
        markers=["o", "s", "^", "d", "x", "*"],
        colors=["red", "green", "blue", "orange", "purple"]
    )

    assert len(table) > 0
    plt.close(fig)


def test_sfr_with_plot_true_and_none_ax_raises():
    with pytest.raises(AttributeError):
        mod.sfr(plot=True, axis=None)


def test_sfr_returns_table():
    table = mod.sfr(plot=False)
    assert len(table) > 0
    assert "Reference" in table.colnames


with pytest.raises(AttributeError):
    mod.sfr(plot=True, axis=None)