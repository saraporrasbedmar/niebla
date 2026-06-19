import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.niebla.measurements_folder.emissivity as mod


def test_emissivity_returns_table():
    data = mod.emissivity(plot=False)

    assert len(data) > 0
    assert "z" in data.colnames
    assert "lambda" in data.colnames
    assert "eje" in data.colnames
    assert "reference" in data.colnames


def test_emissivity_filters_z_range():
    data = mod.emissivity(z_min=1.0, z_max=2.0, plot=False)

    assert np.all(data["z"] >= 1.0)
    assert np.all(data["z"] <= 2.0)


def test_emissivity_filters_lambda_range():
    data = mod.emissivity(lambda_min=0.1, lambda_max=1.0, plot=False)

    assert np.all(data["lambda"] >= 0.1)
    assert np.all(data["lambda"] <= 1.0)


def test_emissivity_filters_references():
    # Use a reference that should exist in the file
    data = mod.emissivity(
        take_only_refs=["Andrews et al. (2017)"], plot=False)

    refs = set(data["reference"].astype(str))
    assert refs == {"Andrews et al. (2017)"}


def test_emissivity_multiple_references():
    data = mod.emissivity(
        take_only_refs=["Andrews et al. (2017)", "Yoshida et al. (2006)"],
        plot=False)

    refs = set(data["reference"].astype(str))
    assert refs.issubset({"Andrews et al. (2017)", "Yoshida et al. (2006)"})


def test_emissivity_plotting_runs():
    fig, ax = plt.subplots()

    data = mod.emissivity(plot=True, axis=ax)

    assert len(data) > 0
    plt.close(fig)


def test_emissivity_plotting_with_legend_runs():
    fig, ax = plt.subplots()

    data = mod.emissivity(plot=True, axis=ax, show_legend=True)

    assert len(data) > 0
    plt.close(fig)