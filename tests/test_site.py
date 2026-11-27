import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
site_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site_build)


def test_builds_from_repo_data(tmp_path):
    site_build.build(ROOT / "data", tmp_path / "out")
    page = (tmp_path / "out" / "index.html").read_text()
    assert "<h1>evosoc</h1>" in page and "Round 0" in page
    assert 'id="baseline_v0"' in page
    assert "{{" not in page
    assert (tmp_path / "out" / "style.css").exists()


def test_escapes_and_lists_matches(tmp_path):
    data = tmp_path / "data"
    (data / "contestants").mkdir(parents=True)
    (data / "matches").mkdir()
    (data / "contestants" / "x_v1.json").write_text(json.dumps({
        "id": "x_v1", "name": "<X>", "version": 1, "team": "centralizers",
        "idea": "a & b", "battle_cry": "go"}))
    (data / "contestants" / "y_v1.json").write_text(json.dumps({
        "id": "y_v1", "name": "Y", "version": 1, "team": "decentralizers", "idea": "y"}))
    (data / "matches" / "001.json").write_text(json.dumps({
        "match": 1, "contestants": {"a": "x_v1", "b": "y_v1", "baseline": "baseline_v1"},
        "winner": "a", "score": "2-1"}))
    site_build.build(data, tmp_path / "out")
    page = (tmp_path / "out" / "index.html").read_text()
    assert "&lt;X&gt;" in page and "a &amp; b" in page
    assert 'id="match-1"' in page and "x_v1 wins" in page and ">1</span>" in page
