"""Unit tests for coruscant plotting helpers."""

import shutil
import unittest

from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import coruscant.plots.plot as plot_module
from coruscant.plots import PlotManager, add_subplot_labels


class PlotTests(unittest.TestCase):
    """Cover the plotting helper utilities with standard-library unittest."""

    def setUp(self) -> None:
        """Reset matplotlib state between tests."""
        plt.rcdefaults()
        plot_module.verify_usetex.cache_clear()

    def tearDown(self) -> None:
        """Clear figure state between tests."""
        plt.close("all")
        plt.rcdefaults()
        plot_module.verify_usetex.cache_clear()

    def test_verify_usetex_reports_missing_latex(self) -> None:
        """The explicit usetex probe should explain why TeX rendering is unavailable."""
        with mock.patch.object(plot_module.shutil, "which", return_value=None):
            available, reason = plot_module.verify_usetex()

        self.assertFalse(available)
        self.assertEqual(reason, "latex executable not found on PATH")

    def test_plot_settings_falls_back_when_usetex_is_unavailable(self) -> None:
        """Explicit usetex requests should downgrade cleanly when validation fails."""
        with mock.patch.object(
            plot_module,
            "verify_usetex",
            return_value=(False, "latex executable not found on PATH"),
        ):
            with self.assertWarnsRegex(UserWarning, "falling back to matplotlib text"):
                plot_module.plot_settings({"text.usetex": True})

        self.assertFalse(plt.rcParams["text.usetex"])

    def test_plot_manager_writes_figures_to_output_dir(self) -> None:
        """PlotManager should save figures to the configured test output directory."""
        output_dir = Path(__file__).resolve().parent / "out" / self._testMethodName
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manager = PlotManager(
            root=str(output_dir),
            overwrite=True,
            custom_settings={"text.usetex": False},
        )

        with manager.make_plot("line_plot") as (fig, ax):
            ax.plot([0.0, 1.0], [1.0, 0.0])
            ax.set_title("line plot")

        self.assertTrue((output_dir / "line_plot.png").exists())

    def test_add_subplot_labels_adds_one_label_per_axis(self) -> None:
        """Each subplot should receive exactly one panel label."""
        fig, axes = plt.subplots(ncols=2)

        add_subplot_labels(axes)

        self.assertEqual([axis.texts[0].get_text() for axis in axes], ["(a)", "(b)"])
        plt.close(fig)