"""Render the evosoc site from the results data.

    python site/build.py [--data data] [--out _site]

Reads data/contestants/*.json, data/matches/*.json and
data/tuning/baseline_v1.json and writes static HTML.
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
    "centralizers": "The Control Room",
    "decentralizers": "Team Decentralizers",
    "neutral": "The House",
}
TICKER_IDLE = [
    "Round 0 is underway",
    "The Baseline is warming up",
    "No matches played yet",
    "Nobody has seen the map",
    "Every guess costs one unit",
    "Stick around",
]
TICKER_ROUND1 = [
    "Round 1 is coming",
    "A challenger has entered",
    "Nobody has seen the map",
    "Every guess costs one unit",
    "Stick around",
]
TICKER_TUNED = [
    "Round 0 is done",
    "The Baseline is ready",
    "Waiting on the first challengers",
    "Nobody has seen the map",
    "Every guess costs one unit",
    "Stick around",
]


def load_dir(path: Path) -> list[dict]:
    if not path.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def contestant_card(c: dict, stage: bool = False) -> str:
    team = c.get("team", "neutral")
    cry = f'<p class="cry">“{esc(c["battle_cry"])}”</p>' if c.get("battle_cry") else ""
    anchor = "" if stage else f' id="{esc(c["id"])}"'
    return (
        f'<article class="card {esc(team)}"{anchor}>'
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


def round0(tuning: dict) -> str:
    """The stage before the first match, once the Baseline is tuned."""
    final = tuning["final"]
    untuned = next(r for r in final if r["settings"]["population"] == 100
                   and r["settings"]["crossover"] == "order_crossover")
    sprint_old = untuned["events"]["sprint"]["mean_gap"]
    sprint_new = final[0]["events"]["sprint"]["mean_gap"]
    tried = tuning["method"]["screen_settings"]
    return (
        '<p class="kicker">Round 0</p>'
        '<h2 class="headline">Meet the Baseline</h2>'
        "<p>Before the teams play, they need something to beat. This is it: a by-the-book "
        "algorithm that belongs to nobody. Thirty trips, the two best are always kept, the rest are "
        "replaced every round by children of strong parents, and every child gets shaken up a little. "
        f"We tried {tried} ways to set it up on practice maps and kept the best. Out of the box it "
        f"finished a quick game {sprint_old:.0f}% off a good trip; tuned, it's {sprint_new:.0f}% off.</p>"
        "<p>Both teams now start from this exact copy. Next up: each one changes who gets to decide, "
        "and we find out whose way works.</p>"
        '<p><a href="https://github.com/ordialascent/evosoc/blob/main/docs/round0.md">'
        "Full tuning record</a></p>"
    )


def challengers(entrants: list[dict]) -> str:
    """The stage before the first match: every team contestant entered so far."""
    waiting = [t for t in ("centralizers", "decentralizers")
               if not any(c["team"] == t for c in entrants)]
    cards = "".join(contestant_card(c, stage=True) for c in entrants)
    if waiting:
        names = " and ".join(f'<strong class="{t[0]}">{esc(TEAMS[t])}</strong>' for t in waiting)
        later = (f"<p>{names} {'is' if len(waiting) == 1 else 'are'} still in the lab. "
                 "Round 1 starts the moment both teams have someone on stage.</p>")
    else:
        later = "<p>Both teams have entered. Round 1 is next: same maps, same budget, same dice.</p>"
    return ('<p class="kicker">Round 1</p>'
            '<h2 class="headline">New challenger!</h2>'
            f"{cards}{later}")


def render(data: Path) -> str:
    contestants = load_dir(data / "contestants")
    # Only the current version of the House is on stage; older ones stay in the logs.
    house = [c for c in contestants if c.get("team", "neutral") == "neutral"]
    if house:
        newest = max(house, key=lambda c: c["version"])
        contestants = [c for c in contestants if c.get("team", "neutral") != "neutral" or c is newest]
    tuning_path = data / "tuning" / "baseline_v1.json"
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else None
    matches = sorted(load_dir(data / "matches"), key=lambda m: m["match"], reverse=True)
    teams = {c["id"]: c.get("team", "neutral") for c in contestants}
    total = score(matches, teams)
    entrants = [c for c in contestants if c.get("team") in ("centralizers", "decentralizers")]

    if matches:
        latest = matches[0]
        stage = (f'<p class="kicker">Last time on evosoc</p>{match_card(latest)}')
        history = "\n".join(match_card(m) for m in matches)
        ticker = [f"Match {latest['match']}: {latest.get('score', '')}",
                  f"{TEAMS['centralizers']} {total['centralizers']}, {TEAMS['decentralizers']} {total['decentralizers']}",
                  "The losers are back in the lab", "Stick around"]
    elif entrants:
        stage = challengers(entrants)
        history = '<p class="empty">No matches yet. The first one is close.</p>'
        ticker = TICKER_ROUND1
    elif tuning:
        stage = round0(tuning)
        history = '<p class="empty">No matches yet. The teams are in the lab.</p>'
        ticker = TICKER_TUNED
    else:
        stage = (
            '<p class="kicker">Tonight: Round 0</p>'
            '<h2 class="headline">Meet the House</h2>'
            '<p>Before the teams play, they need something to beat. The Baseline is a '
            'by-the-book algorithm that doesn\'t belong to either team. We\'re tuning it right now. '
            'Both teams start out as copies of it, then each one makes it their own.</p>'
        )
        history = '<p class="empty">No matches yet. Give it a minute.</p>'
        ticker = TICKER_IDLE
    ticker_html = "".join(f"<span>{esc(t)}</span>" for t in ticker * 2)

    page = (HERE / "template.html").read_text()
    for key, value in {
        "team_c": esc(TEAMS["centralizers"]),
        "team_d": esc(TEAMS["decentralizers"]),
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
