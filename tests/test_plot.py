"""Tests for coruscant plotting helpers."""

import shutil
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import coruscant.plots.plot as plot_module
from coruscant.plots import PlotManager, add_subplot_labels


@pytest.fixture(autouse=True)
def reset_matplotlib_state() -> None:
    """Reset matplotlib state between tests."""
    plt.rcdefaults()
    plot_module.verify_usetex.cache_clear()
    yield
    plt.close("all")
    plt.rcdefaults()
    plot_module.verify_usetex.cache_clear()


def test_verify_usetex_reports_missing_latex() -> None:
    """The explicit usetex probe should explain why TeX rendering is unavailable."""
    with mock.patch.object(plot_module.shutil, "which", return_value=None):
        available, reason = plot_module.verify_usetex()

    assert not available
    assert reason == "latex executable not found on PATH"


def test_plot_settings_falls_back_when_usetex_is_unavailable() -> None:
    """Explicit usetex requests should downgrade cleanly when validation fails."""
    with mock.patch.object(
        plot_module,
        "verify_usetex",
        return_value=(False, "latex executable not found on PATH"),
    ):
        with pytest.warns(UserWarning, match="falling back to matplotlib text"):
            plot_module.plot_settings({"text.usetex": True})

    assert not plt.rcParams["text.usetex"]


def test_plot_manager_writes_figures_to_output_dir() -> None:
    """PlotManager should save figures to the configured test output directory."""
    output_dir = Path(__file__).resolve().parent / "out" / "test_plot_manager_writes_figures_to_output_dir"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = PlotManager(
        root=str(output_dir),
        overwrite=True,
        custom_settings={"text.usetex": False},
    )

    with manager.make_plot("line_plot") as (_, ax):
        ax.plot([0.0, 1.0], [1.0, 0.0])
        ax.set_title("line plot")

    assert (output_dir / "line_plot.png").exists()


def test_add_subplot_labels_adds_one_label_per_axis() -> None:
    """Each subplot should receive exactly one panel label."""
    fig, axes = plt.subplots(ncols=2)

    add_subplot_labels(axes)

    assert [axis.texts[0].get_text() for axis in axes] == ["(a)", "(b)"]
    plt.close(fig)
