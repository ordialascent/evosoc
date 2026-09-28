import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evosoc.analysis import analyze
from evosoc.analysis.stats import a12, holm, verdict
from evosoc.experiment import ExperimentConfig, load_runs, run_experiment

SMOKE = Path(__file__).parents[1] / "configs" / "smoke.toml"


@pytest.fixture
def cfg(tmp_path):
    """The smoke config, shrunk so the tests stay fast."""
    text = (
        SMOKE.read_text()
        .replace("budget = 5_000", "budget = 1_000")
        .replace("seeds = 10", "seeds = 6")
    )
    path = tmp_path / "smoke.toml"
    path.write_text(text)
    return ExperimentConfig.from_toml(path)


def test_run_and_analyze(cfg, tmp_path):
    exp_dir = run_experiment(cfg, tmp_path, jobs=2)
    df = load_runs(exp_dir)

    runs = df.groupby(["variant", "problem", "seed"])
    assert runs.ngroups == len(cfg.variants) * len(cfg.problems) * len(cfg.seeds)
    # Every run ends exactly at the budget and best-so-far never gets worse.
    assert (runs["evaluations"].max() == cfg.budget).all()
    assert runs["best_so_far"].apply(lambda s: s.is_monotonic_decreasing).all()

    imp = load_runs(exp_dir, "improvements").groupby(["variant", "problem", "seed"])
    assert (imp["evaluations"].min() == 1).all()  # first evaluation is always an improvement
    assert (imp["best_so_far"].min() == runs["best_so_far"].min()).all()

    report = analyze(exp_dir)
    assert (report / "figures" / "convergence.png").exists()
    assert (report / "summary.md").exists()


def test_rerun_is_resumed_and_deterministic(cfg, tmp_path):
    exp_dir = run_experiment(cfg, tmp_path, jobs=1)
    before = load_runs(exp_dir).drop(columns="time_s")
    before_imp = load_runs(exp_dir, "improvements")
    shutil.rmtree(exp_dir / "runs" / "random-25")
    run_experiment(cfg, tmp_path, jobs=2)
    pd.testing.assert_frame_equal(before, load_runs(exp_dir).drop(columns="time_s"))
    pd.testing.assert_frame_equal(before_imp, load_runs(exp_dir, "improvements"))


def test_config_errors_surface_at_load(tmp_path):
    bad = SMOKE.read_text().replace("params = { pop_size = 50 }", "params = { pop_szie = 50 }")
    path = tmp_path / "bad.toml"
    path.write_text(bad)
    with pytest.raises(ValueError, match="variant 'ga'"):
        ExperimentConfig.from_toml(path)


def test_a12():
    assert a12(np.array([1, 2]), np.array([3, 4])) == 1.0
    assert a12(np.array([3, 4]), np.array([1, 2])) == 0.0
    assert a12(np.array([1, 2]), np.array([1, 2])) == 0.5


def test_holm():
    np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_verdict():
    assert verdict(0.8, 0.01) == "better"
    assert verdict(0.2, 0.01) == "worse"
    assert verdict(0.8, 0.2) == "no clear difference"


def test_analyze_works_after_a_variants_code_is_gone(cfg, tmp_path):
    exp_dir = run_experiment(cfg, tmp_path, jobs=2)
    config = exp_dir / "config.toml"
    config.write_text(config.read_text().replace('"steady_state_ga"', '"deleted_ga"'))
    assert (analyze(exp_dir) / "summary.md").exists()
