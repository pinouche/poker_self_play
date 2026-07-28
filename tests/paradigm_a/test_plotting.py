"""Training-metric plotting utilities."""

import json
import os
import tempfile

import pytest

from paradigm_a.training.plotting import (
    load_history,
    plot_runs,
    run_label,
    series,
    smooth,
    summarize,
)


def fake_history(n=30, offset=0.0):
    """Early iterations have no training metrics: the buffer is still filling."""
    out = []
    for i in range(1, n + 1):
        record = {"iteration": i, "showdown_rate": 0.5}
        if i > 5:
            record.update(entropy=1.0 + offset, loss=2.0 - i * 0.01, q_loss=0.2)
        out.append(record)
    return out


def test_series_skips_iterations_without_the_metric():
    xs, ys = series(fake_history(), "entropy")
    assert xs[0] == 6 and len(xs) == 25
    assert len(xs) == len(ys)


def test_series_is_empty_for_an_unknown_metric():
    assert series(fake_history(), "not_a_metric") == ([], [])


def test_smoothing_preserves_length_and_reduces_variance():
    import numpy as np

    values = list(np.random.default_rng(0).normal(0, 1, 100))
    smoothed = smooth(values, 9)
    assert len(smoothed) == len(values)
    assert np.std(smoothed) < np.std(values)


def test_smoothing_is_a_noop_for_tiny_windows():
    assert smooth([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]


def test_run_label_uses_the_checkpoint_directory_name():
    assert run_label("/a/b/runA") == "runA"
    assert run_label("/a/b/runA/") == "runA"
    assert run_label("/a/b/runA/history.json") == "runA"


def test_load_history_accepts_a_directory_or_a_file():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "history.json")
        with open(path, "w") as fh:
            json.dump(fake_history(), fh)
        assert len(load_history(directory)) == 30
        assert len(load_history(path)) == 30


def test_plot_runs_builds_a_figure_for_several_runs():
    figure = plot_runs(
        {"a": fake_history(offset=0.0), "b": fake_history(offset=0.3)},
        metrics=["entropy", "loss"],
        window=5,
    )
    assert len(figure.axes) >= 2
    entropy_axis = figure.axes[0]
    assert entropy_axis.get_title() == "entropy"
    # Two runs plus the "uniform" reference line.
    assert len(entropy_axis.lines) >= 2


def test_plot_runs_writes_a_file():
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "plot.png")
        plot_runs({"a": fake_history()}, metrics=["entropy"], out=out)
        assert os.path.getsize(out) > 0


def decaying_history(n=30, floor=0.0):
    """``q_loss`` falling by two orders of magnitude, as it does in a real run."""
    return [
        {"iteration": i, "q_loss": floor + 0.5 * (0.85**i)} for i in range(1, n + 1)
    ]


def test_decaying_metrics_get_a_log_y_axis():
    axis = plot_runs({"a": decaying_history()}, metrics=["q_loss"]).axes[0]
    assert axis.get_yscale() == "log"
    assert axis.get_title() == "q_loss (log scale)"


def test_a_non_positive_sample_falls_back_to_a_linear_axis():
    """A log axis drops non-positive points silently, which would hide data."""
    history = decaying_history()
    history[3]["q_loss"] = 0.0
    axis = plot_runs({"a": history}, metrics=["q_loss"]).axes[0]
    assert axis.get_yscale() == "linear"


def test_other_metrics_keep_their_linear_axis():
    axis = plot_runs({"a": fake_history()}, metrics=["entropy"]).axes[0]
    assert axis.get_yscale() == "linear"


def test_plot_runs_rejects_metrics_that_appear_nowhere():
    with pytest.raises(ValueError):
        plot_runs({"a": fake_history()}, metrics=["nonexistent"])


def test_summarize_reports_a_row_per_run():
    with tempfile.TemporaryDirectory() as directory:
        for name, offset in (("runA", 0.0), ("runB", 0.5)):
            os.makedirs(os.path.join(directory, name))
            with open(os.path.join(directory, name, "history.json"), "w") as fh:
                json.dump(fake_history(offset=offset), fh)
        text = summarize(
            [os.path.join(directory, n) for n in ("runA", "runB")], ["entropy"]
        )
    assert "runA" in text and "runB" in text
    assert "1.0000" in text and "1.5000" in text
