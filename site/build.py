"""Render the evosoc site from the results data.

    python site/build.py [--data data] [--out _site]

Reads data/contestants/*.json and data/matches/*.json and writes static HTML.
No dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEAMS = {
    "centralizers": "Team Centralizers",
    "decentralizers": "Team Decentralizers",
    "neutral": "Neutral",
}


def load_dir(path: Path) -> list[dict]:
    if not path.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def contestant_card(c: dict) -> str:
    team = c.get("team", "neutral")
    cry = f'<p class="cry">“{esc(c["battle_cry"])}”</p>' if c.get("battle_cry") else ""
    return (
        f'<article class="card {esc(team)}" id="{esc(c["id"])}">'
        f'<p class="team">{esc(TEAMS.get(team, team))}</p>'
        f'<h3>{esc(c["name"])} <span class="ver">v{esc(c["version"])}</span></h3>'
        f'<p>{esc(c["idea"])}</p>{cry}</article>'
    )


def match_card(m: dict) -> str:
    names = m["contestants"]
    winner = m.get("winner")
    result = f'Winner: <strong>{esc(names[winner])}</strong>' if winner else "Draw"
    return (
        f'<article class="card match" id="match-{esc(m["match"])}">'
        f'<h3>Match {esc(m["match"])}: {esc(names["a"])} vs {esc(names["b"])}</h3>'
        f'<p>{result} · {esc(m.get("score", ""))}</p></article>'
    )


def render(data: Path) -> str:
    contestants = load_dir(data / "contestants")
    matches = sorted(load_dir(data / "matches"), key=lambda m: m["match"], reverse=True)
    if matches:
        history = "\n".join(match_card(m) for m in matches)
    else:
        history = '<p class="empty">No matches yet. The first bell rings after Round 1.</p>'
    cast = "\n".join(contestant_card(c) for c in contestants)
    page = (HERE / "template.html").read_text()
    return page.replace("{{contestants}}", cast).replace("{{matches}}", history)


def build(data: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "index.html").write_text(render(data))
    shutil.copy(HERE / "style.css", out / "style.css")
    (out / ".nojekyll").write_text("")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("_site"))
    args = parser.parse_args(argv)
    build(args.data, args.out)
    print(f"site written to {args.out}/")


if __name__ == "__main__":
    main()
