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
    "neutral": "The House",
}
TICKER_IDLE = [
    "Round 0 in progress",
    "The Baseline is training",
    "No matches have been played",
    "Nobody has seen the map",
    "Every measured trip costs one unit",
    "Please stay tuned",
]


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
    if winner:
        loser = "b" if winner == "a" else "a"
        result = (f'<p class="winner">{esc(names[winner])} wins</p>'
                  f'<p class="loser">over {esc(names[loser])}</p>')
    else:
        result = f'<p class="winner">Draw</p><p class="loser">{esc(names["a"])} vs {esc(names["b"])}</p>'
    takeaway = f'<p class="takeaway">{esc(m["takeaway"])}</p>' if m.get("takeaway") else ""
    return (
        f'<article class="card match" id="match-{esc(m["match"])}">'
        f'<p class="team">Match {esc(m["match"])} · {esc(m.get("score", ""))}</p>'
        f'{result}{takeaway}</article>'
    )


def score(matches: list[dict], teams: dict[str, str]) -> dict[str, int]:
    total = {"centralizers": 0, "decentralizers": 0}
    for m in matches:
        w = m.get("winner")
        if w and teams.get(m["contestants"][w]) in total:
            total[teams[m["contestants"][w]]] += 1
    return total


def render(data: Path) -> str:
    contestants = load_dir(data / "contestants")
    matches = sorted(load_dir(data / "matches"), key=lambda m: m["match"], reverse=True)
    teams = {c["id"]: c.get("team", "neutral") for c in contestants}
    total = score(matches, teams)

    if matches:
        latest = matches[0]
        stage = (f'<p class="kicker">Last time on evosoc</p>{match_card(latest)}')
        history = "\n".join(match_card(m) for m in matches)
        ticker = [f"Match {latest['match']}: {latest.get('score', '')}",
                  f"Centralizers {total['centralizers']}, Decentralizers {total['decentralizers']}",
                  "The loser is already back in the lab", "Please stay tuned"]
    else:
        stage = (
            '<p class="kicker">Tonight</p>'
            '<h2 class="headline">Round 0: meet the House.</h2>'
            '<p>Before anyone fights, we need a yardstick. The Baseline is a plain, '
            'by-the-book algorithm with no opinions about anything. It is being tuned right now. '
            'Both teams will be built from it. Then they will try to destroy it.</p>'
        )
        history = '<p class="empty">Nothing has happened yet. This is the calm before.</p>'
        ticker = TICKER_IDLE
    ticker_html = "".join(f"<span>{esc(t)}</span>" for t in ticker * 2)

    page = (HERE / "template.html").read_text()
    for key, value in {
        "stage": stage,
        "score_c": str(total["centralizers"]),
        "score_d": str(total["decentralizers"]),
        "ticker": ticker_html,
        "contestants": "\n".join(contestant_card(c) for c in contestants),
        "matches": history,
    }.items():
        page = page.replace("{{" + key + "}}", value)
    return page


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
