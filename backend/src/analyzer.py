import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io
import base64
import math
import random
import statistics
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# Minimal dataclasses and parsers to support CSV-first analysis when full JSON schema is absent
@dataclass
class MatchSummary:
    teams: List[str] = field(default_factory=list)
    match_type: str = ""
    winner: Optional[str] = None
    player_of_match: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    city: Optional[str] = None
    dates: List[str] = field(default_factory=list)
    result_by: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamInningsSummary:
    team: str = ""
    innings_index: int = 1
    overs_bowled: int = 0
    deliveries_counted: int = 0
    declared: bool = False


@dataclass
class Insights:
    match: MatchSummary
    innings: List[TeamInningsSummary]
    key_points: List[str]
    tactical_notes: List[str]
    csv_summary: Optional[Dict[str, Any]] = None
    plots: Optional[Dict[str, str]] = None
    monte: Optional[Dict[str, Any]] = None
    csv_only: bool = False


def parse_match_summary(raw: Dict[str, Any]) -> MatchSummary:
    info = raw.get('info', {}) if isinstance(raw, dict) else {}
    teams = info.get('teams') or []
    match_type = info.get('match_type') or info.get('type') or ''
    winner = info.get('outcome', {}).get('winner') if info.get('outcome') else info.get('winner')
    pom = info.get('player_of_match') or info.get('player_of_the_match') or []
    venue = info.get('venue') or info.get('ground')
    city = info.get('city')
    dates = info.get('dates') or info.get('date') and [info.get('date')] or []
    result_by = info.get('result_by') or {}
    return MatchSummary(teams=teams, match_type=match_type or '', winner=winner, player_of_match=pom if isinstance(pom, list) else [pom] if pom else [], venue=venue, city=city, dates=dates or [], result_by=result_by)


def summarize_innings(raw: Dict[str, Any]) -> List[TeamInningsSummary]:
    # Best-effort: if raw contains innings summaries, convert them; else return empty list
    inns = []
    raw_inns = raw.get('innings') if isinstance(raw, dict) else None
    if isinstance(raw_inns, list):
        for idx, r in enumerate(raw_inns, start=1):
            team = r.get('team') or r.get('name') or f'Team {idx}'
            overs = r.get('overs') or r.get('overs_bowled') or 0
            delivs = r.get('deliveries_counted') or 0
            declared = bool(r.get('declared'))
            inns.append(TeamInningsSummary(team=team, innings_index=idx, overs_bowled=overs, deliveries_counted=delivs, declared=declared))
    return inns


def derive_key_points(ms: 'MatchSummary', inn: List['TeamInningsSummary']) -> List[str]:
    """Extract short bullet key points from a MatchSummary and innings list."""
    points: List[str] = []
    if getattr(ms, 'winner', None):
        by_txt = (
            f" by {next(iter(ms.result_by))} {next(iter(ms.result_by.values()))}" if getattr(ms, 'result_by', None) else ""
        )
        points.append(f"Result: {ms.winner}{by_txt}.")
    if getattr(ms, 'player_of_match', None):
        pom = ms.player_of_match
        if isinstance(pom, (list, tuple)):
            points.append("Player of the Match: " + ", ".join(pom))
        else:
            points.append(f"Player of the Match: {pom}")
    if getattr(ms, 'venue', None) or getattr(ms, 'city', None):
        loc = ", ".join([p for p in [getattr(ms, 'venue', None), getattr(ms, 'city', None)] if p])
        if loc:
            points.append(f"Location: {loc}.")
    if getattr(ms, 'dates', None):
        try:
            # lazy import for date parsing
            try:
                import dateutil.parser as dateparser
            except Exception:
                dateparser = None
            if dateparser and ms.dates:
                start = dateparser.parse(ms.dates[0]).date()
                end = dateparser.parse(ms.dates[-1]).date()
                date_str = str(start) if start == end else f"{start} to {end}"
                points.append(f"Dates: {date_str}.")
            else:
                points.append(f"Dates: {', '.join(ms.dates)}.")
        except Exception:
            points.append(f"Dates: {', '.join(ms.dates)}.")
    if getattr(ms, 'teams', None):
        teams = ms.teams
        if isinstance(teams, (list, tuple)) and len(teams) >= 2:
            points.append(f"Teams: {teams[0]} vs {teams[1]}.")
    # Innings completeness (if innings summaries provided)
    for s in inn or []:
        tag = f"{getattr(s, 'team', 'Team')} (Innings {getattr(s, 'innings_index', '?')})"
        points.append(
            f"{tag}: recorded overs={getattr(s, 'overs_bowled', 'N')}, deliveries listed={getattr(s, 'deliveries_counted', 'N')}, declared={getattr(s, 'declared', False)}."
        )
    return points


def derive_tactical_notes(ms: 'MatchSummary', inn: List['TeamInningsSummary']) -> List[str]:
    notes: List[str] = []
    # If the analysis is based on JSON (raw_data from JSON), warn about truncation; otherwise skip
    if globals().get('RAW_FROM_JSON', False):
        notes.append(
            "Data is partially truncated (many deliveries omitted), so run rates and phase-by-phase breakdowns are approximate or unavailable."
        )
    # Toss impact
    toss = raw_data.get('info', {}).get('toss') if 'raw_data' in globals() else {}
    if isinstance(toss, dict):
        decision = toss.get("decision")
        winner = toss.get("winner")
        if decision and winner:
            notes.append(f"Toss: {winner} chose to {decision}. Consider adjusting early tactics accordingly.")
    # Suggest generic Test tactics
    match_type = getattr(ms, 'match_type', '') or ''
    if isinstance(match_type, str) and match_type.lower() == "test":
        notes.extend([
            "For Tests, prioritize building pressure: attacking fields early with a new ball; conserve bowler workloads for optimal spells.",
            "Use left-right batting combinations to disrupt bowler rhythm; rotate strike to avoid maidens stacking.",
            "Track partnership stability and bowler spell effectiveness once full data is available.",
        ])
    return notes


def analyze_match(json_path: Path) -> Insights:
    global raw_data
    raw_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    ms = parse_match_summary(raw_data)
    inn = summarize_innings(raw_data)
    kp = derive_key_points(ms, inn)
    tn = derive_tactical_notes(ms, inn)
    # Attempt CSV-based analytics if deliveries.csv exists next to json or in repo root
    csv_summary: Optional[Dict[str, Any]] = None
    # Look for deliveries.csv in same folder and repo root
    candidates = [Path("deliveries.csv"), Path(json_path).resolve().parent / "deliveries.csv"]
    for c in candidates:
        if c.exists():
            try:
                deliveries = load_deliveries_csv(c)
                csv_summary = {
                    "phase_run_rates": compute_phase_run_rates(deliveries, raw_data),
                    "wicket_clusters": compute_wicket_clusters(deliveries, raw_data),
                }
            except Exception:
                csv_summary = None
            break

    return Insights(match=ms, innings=inn, key_points=kp, tactical_notes=tn, csv_summary=csv_summary)


def insights_to_markdown(ins: Insights) -> str:
    ms = ins.match
    lines: List[str] = []
    title = f"Strategist Insights: {ms.teams[0]} vs {ms.teams[1]} ({ms.match_type})" if ms.teams else "Strategist Insights"
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Summary")
    for p in ins.key_points:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## Tactical Notes")
    for n in ins.tactical_notes:
        lines.append(f"- {n}")
    lines.append("")
    # CSV analytics section
    if ins.csv_summary:
        lines.append("## CSV Analytics")
        pr = ins.csv_summary.get("phase_run_rates", {})
        if pr:
            lines.append("")
            lines.append("### Phase-wise Run Rates (runs per over)")
            for phase_name, stats in pr.items():
                runs = stats.get("runs", 0)
                balls = stats.get("balls", 0)
                rpo = stats.get("rpo", 0)
                lines.append(f"- {phase_name}: runs={runs}, balls={balls}, rpo={rpo:.2f}")
        # per-team phase run rates
        pt = ins.csv_summary.get('phase_run_rates_per_team', {})
        if pt:
            lines.append("")
            lines.append("### Per-team Phase-wise Run Rates")
            for team, phases in pt.items():
                lines.append(f"- {team}:")
                for pname, stats in phases.items():
                    lines.append(f"  - {pname}: runs={stats['runs']}, balls={stats['balls']}, rpo={stats['rpo']:.2f}")
        wc = ins.csv_summary.get("wicket_clusters", [])
        if wc:
            lines.append("")
            lines.append("### Wicket Clusters (top windows)")
            for w in wc:
                lines.append(f"- Innings {w['innings']}: overs {w['start_over']}–{w['end_over']} => wickets={w['wickets']}")
        bw = ins.csv_summary.get('bowler_wicket_clusters', [])
        if bw:
            lines.append("")
            lines.append("### Bowler Wicket Clusters (top windows)")
            for b in bw[:10]:
                lines.append(f"- {b.get('bowler_name')} (id={b.get('bowler_id')}), Innings {b['innings']}: overs {b['start_over']}–{b['end_over']} => wickets={b['wickets']}")
        lines.append("")
    # Plots
    if ins.plots:
        lines.append("## Plots")
        for k, v in (ins.plots or {}).items():
            if not v:
                continue
            # If value is a data URI embed it inline, otherwise treat as a file path (relative to workspace)
            if isinstance(v, str) and v.startswith('data:image'):
                lines.append(f"### {k}")
                lines.append("")
                lines.append(f"![{k}]({v})")
                lines.append("")
            else:
                # normalize the path string for Markdown (use forward slashes)
                path_str = str(v).replace('\\', '/')
                lines.append(f"### {k}")
                lines.append("")
                lines.append(f"![{k}]({path_str})")
                lines.append("")
    # Monte Carlo
    if ins.monte:
        lines.append("## Monte Carlo Simulation")
        for kk, vv in ins.monte.items():
            lines.append(f"- {kk}: {vv}")
        lines.append("")
    lines.append("## Data Completeness")
    if getattr(ins, 'csv_only', False):
        lines.append("- Analysis used CSV files (deliveries.csv / matches.csv). CSV completeness may vary; computations use all available CSV rows.")
    else:
        lines.append("- Deliveries are truncated in the provided JSON; some overs contain placeholders. Computations are conservative.")
    return "\n".join(lines)


# ---------------- CSV Analytics Helpers ----------------
def load_deliveries_csv(path: Path) -> List[Dict[str, Any]]:
    import csv

    deliveries: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize numeric fields
            # safer conversion pattern
            mid = r.get('match_id')
            if mid is None or mid == '':
                r['match_id'] = None
            else:
                try:
                    r['match_id'] = int(mid)
                except Exception:
                    r['match_id'] = None

            for k in ('innings', 'over', 'ball', 'runs_off_bat', 'extras', 'delivery_id'):
                val = r.get(k)
                if val is None or val == '':
                    r[k] = None
                else:
                    try:
                        r[k] = int(val)
                    except Exception:
                        r[k] = None
            # wicket_type/dismissed_batter_id may be empty strings
            if 'dismissed_batter_id' in r:
                db = r.get('dismissed_batter_id')
                if db in (None, '', 'None'):
                    r['dismissed_batter_id'] = None
                else:
                    try:
                        r['dismissed_batter_id'] = int(db)
                    except Exception:
                        r['dismissed_batter_id'] = None
            deliveries.append(r)
    return deliveries


def compute_phase_run_rates(deliveries: List[Dict[str, Any]], raw_json: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Compute simple phase-wise run rates for a match present in raw_json.
    Phases (default): Powerplay (overs 0-5), Middle (6-40), Late (41+).
    Returns dict of {phase: {runs, balls, rpo}}
    """
    # Try to get match identifier (if available in raw_json.info.match_id or registry)
    match_meta = raw_json.get('info', {})
    # We will try to match by teams if match_id isn't in CSV
    teams = match_meta.get('teams', [])

    # Determine candidate match_ids from deliveries matching teams if possible
    # Simpler: process all deliveries but filter by match_id when available in JSON 'match_id' field
    # If raw_json has no numeric match_id, attempt to match by teams and date (best-effort)
    match_id = None
    if isinstance(raw_json.get('meta', {}).get('match_id'), int):
        match_id = raw_json['meta']['match_id']

    # If no numeric match_id, attempt to find any deliveries that include either of the teams by team ids is not available.
    # For PoC, when match_id is None we'll compute phases across all deliveries for simplicity.

    phases = {
        'Powerplay (0-5)': (0, 5),
        'Middle (6-40)': (6, 40),
        'Late (41+)': (41, 1000),
    }

    result: Dict[str, Dict[str, float]] = {}
    for pname, (start_ov, end_ov) in phases.items():
        runs = 0
        balls = 0
        for d in deliveries:
            # If match_id is present in CSV and JSON, filter
            if match_id is not None and d.get('match_id') != match_id:
                continue
            ov = d.get('over')
            if ov is None:
                continue
            if start_ov <= ov <= end_ov:
                r = d.get('runs_off_bat') or 0
                ex = d.get('extras') or 0
                runs += (r + ex)
                balls += 1
        rpo = (runs / (balls / 6)) if balls > 0 else 0.0
        result[pname] = {'runs': runs, 'balls': balls, 'rpo': round(rpo, 3)}
    return result


def compute_wicket_clusters(deliveries: List[Dict[str, Any]], raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute wicket clusters using a sliding window of 6 overs within each innings.
    Returns top 3 windows with highest wickets: list of {innings, start_over, end_over, wickets}
    """
    # Build mapping: innings -> over -> wicket count
    window_size = 6
    match_id = None
    if isinstance(raw_json.get('meta', {}).get('match_id'), int):
        match_id = raw_json['meta']['match_id']

    innings_map: Dict[int, Dict[int, int]] = {}
    for d in deliveries:
        if match_id is not None and d.get('match_id') != match_id:
            continue
        inn = d.get('innings') or 1
        over = d.get('over')
        if over is None:
            continue
        w = 0
        if d.get('dismissed_batter_id'):
            w = 1
        innings_map.setdefault(inn, {})
        innings_map[inn][over] = innings_map[inn].get(over, 0) + w

    clusters: List[Dict[str, Any]] = []
    for inn, over_map in innings_map.items():
        overs_sorted = sorted(over_map.keys())
        if not overs_sorted:
            continue
        max_over = max(overs_sorted)
        # slide window
        for start in range(0, max_over + 1):
            end = start + window_size - 1
            wickets = sum(over_map.get(o, 0) for o in range(start, end + 1))
            clusters.append({'innings': inn, 'start_over': start, 'end_over': end, 'wickets': wickets})

    # sort by wickets desc and return top 3
    clusters_sorted = sorted(clusters, key=lambda x: x['wickets'], reverse=True)
    top = [c for c in clusters_sorted if c['wickets'] > 0][:3]
    return top


def load_players_csv(path: Path = Path("players.csv")) -> Dict[int, str]:
    """Load players.csv to map player_id -> player_name. Returns empty dict if missing."""
    import csv
    mapping: Dict[int, str] = {}
    if not Path(path).exists():
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r.get('player_id') or r.get('id')
            name = r.get('name') or r.get('player_name') or r.get('full_name')
            if not pid:
                continue
            try:
                pid_i = int(pid)
            except Exception:
                continue
            mapping[pid_i] = name or str(pid_i)
    return mapping


def safe_int(val: Any) -> Optional[int]:
    try:
        if val is None or val == '':
            return None
        return int(val)
    except Exception:
        return None


def load_matches_csv(path: Path = Path("matches.csv")) -> List[Dict[str, Any]]:
    """Load matches.csv into a list of dictionaries. Returns empty list if file missing."""
    import csv
    records: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return records
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def derive_match_id_from_csv(deliveries: List[Dict[str, Any]], raw_json: Dict[str, Any]) -> Optional[int]:
    """Best-effort: if deliveries contain match_id, return the most common one that matches the teams/date in raw_json.
    Otherwise try to consult matches.csv (if present) to find a match_id by matching team names and dates.
    Returns None when no confident match is found.
    """
    # First try direct match_id in deliveries
    mids = [d.get('match_id') for d in deliveries if d.get('match_id') is not None]
    if mids:
        # prefer the most common numeric match_id
        try:
            mid = max(set(mids), key=mids.count)
            mid_i = safe_int(mid)
            if mid_i is not None:
                return mid_i
        except Exception:
            pass

    # Try matches.csv if available
    matches_path = Path("matches.csv")
    info = raw_json.get('info', {})
    teams = set(info.get('teams', []))
    dates = [str(d) for d in info.get('dates', [])]
    if matches_path.exists() and teams:
        import csv
        candidates: List[Tuple[int, str, str, str]] = []  # (match_id, team_a, team_b, date)
        with open(matches_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                mid = safe_int(r.get('match_id'))
                if mid is None:
                    continue
                a = (r.get('team_a') or "").strip()
                b = (r.get('team_b') or "").strip()
                date = (r.get('date') or r.get('match_date') or "").strip()
                candidates.append((mid, a, b, date))
        # Try to match by teams and date
        for mid, a, b, date in candidates:
            if teams == {a, b} or teams == {b, a}:
                # If dates overlap or date matches one of raw_json dates, accept
                if not dates or date in dates:
                    return mid
        # fallback: if a teams-only match exists, pick the first
        for mid, a, b, date in candidates:
            if teams == {a, b} or teams == {b, a}:
                return mid

    return None


def compute_phase_run_rates_per_team(deliveries: List[Dict[str, Any]], raw_json: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute phase run rates for each team in the match. Returns {team: {phase: {runs,balls,rpo}}}.
    Heuristic: innings 1 -> team_a, innings 2 -> team_b when matches.csv is available. If not available, falls back to per-innings numbering.
    """
    # derive match-scoped match_id
    match_id = derive_match_id_from_csv(deliveries, raw_json)
    # load match teams
    info = raw_json.get('info', {})
    teams = info.get('teams', [])

    # Determine innings->team mapping heuristically using matches.csv
    inning_team: Dict[int, str] = {}
    matches_path = Path("matches.csv")
    if matches_path.exists() and teams:
        # assume innings 1 -> team_a, innings 2 -> team_b
        import csv
        with open(matches_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                mid = safe_int(r.get('match_id'))
                if mid is None:
                    continue
                if match_id is not None and mid != match_id:
                    continue
                a = r.get('team_a')
                b = r.get('team_b')
                if a:
                    inning_team[1] = a
                if b:
                    inning_team[2] = b
                break

    # default mapping: innings 1/2 map to teams list order if available
    if not inning_team and teams:
        if len(teams) >= 1:
            inning_team[1] = teams[0]
        if len(teams) >= 2:
            inning_team[2] = teams[1]

    phases = {
        'Powerplay (0-5)': (0, 5),
        'Middle (6-40)': (6, 40),
        'Late (41+)': (41, 1000),
    }

    team_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for inn_idx in set([d.get('innings') or 1 for d in deliveries]):
        team = inning_team.get(inn_idx, f'Innings {inn_idx}')
        team_stats.setdefault(team, {})
        for pname, (start_ov, end_ov) in phases.items():
            runs = 0
            balls = 0
            for d in deliveries:
                if match_id is not None and d.get('match_id') != match_id:
                    continue
                if (d.get('innings') or 1) != inn_idx:
                    continue
                ov = d.get('over')
                if ov is None:
                    continue
                if start_ov <= ov <= end_ov:
                    runs += (d.get('runs_off_bat') or 0) + (d.get('extras') or 0)
                    balls += 1
            rpo = (runs / (balls / 6)) if balls > 0 else 0.0
            team_stats[team][pname] = {'runs': runs, 'balls': balls, 'rpo': round(rpo, 3)}

    return team_stats


def compute_per_bowler_wicket_clusters(deliveries: List[Dict[str, Any]], raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute top wicket clusters per bowler across innings using a 6-over sliding window. Returns list of clusters with bowler_id and bowler_name when available."""
    match_id = derive_match_id_from_csv(deliveries, raw_json)
    players = load_players_csv()

    # Build mapping: bowler -> innings -> over -> wickets
    bowler_map: Dict[int, Dict[int, Dict[int, int]]] = {}
    for d in deliveries:
        if match_id is not None and d.get('match_id') != match_id:
            continue
        bow = d.get('bowler_id')
        if not bow:
            continue
        inn = d.get('innings') or 1
        ov = d.get('over')
        if ov is None:
            continue
        w = 1 if d.get('dismissed_batter_id') else 0
        bowler_map.setdefault(bow, {})
        bowler_map[bow].setdefault(inn, {})
        bowler_map[bow][inn][ov] = bowler_map[bow][inn].get(ov, 0) + w

    clusters: List[Dict[str, Any]] = []
    window_size = 6
    for bow, inn_map in bowler_map.items():
        for inn, over_map in inn_map.items():
            if not over_map:
                continue
            max_over = max(over_map.keys())
            for start in range(0, max_over + 1):
                end = start + window_size - 1
                wickets = sum(over_map.get(o, 0) for o in range(start, end + 1))
                if wickets > 0:
                    clusters.append({
                        'bowler_id': bow,
                        'bowler_name': players.get(bow, str(bow)),
                        'innings': inn,
                        'start_over': start,
                        'end_over': end,
                        'wickets': wickets,
                    })

    clusters_sorted = sorted(clusters, key=lambda x: x['wickets'], reverse=True)
    return clusters_sorted[:10]


def plot_phase_run_rates_per_team(team_stats: Dict[str, Dict[str, Dict[str, float]]], out_path: Optional[Path] = None) -> Optional[Any]:
    """Plot phase run rates per team.
    If out_path is provided the PNG is written to disk and the function returns the path (as a string) to embed in Markdown.
    If out_path is None the function falls back to returning a base64 data URI (legacy behavior).
    Returns None if plotting unavailable or no data.
    """
    if plt is None:
        return None
    if not team_stats:
        return None
    teams = list(team_stats.keys())
    phases = list(next(iter(team_stats.values())).keys()) if teams else []
    if not phases:
        return None
    x = range(len(phases))
    # Annotated plotting: clear labels, grid, markers, and annotate points
    fig, ax = plt.subplots(figsize=(max(6, len(phases) * 1.5), 5), dpi=150)
    colors = plt.cm.get_cmap('tab10')
    for idx, team in enumerate(teams):
        rpos = [team_stats[team].get(p, {}).get('rpo', 0.0) for p in phases]
        ax.plot(list(x), rpos, marker='o', label=team, color=colors(idx % 10))
        # annotate each point with its value
        for xi, yi in zip(x, rpos):
            try:
                ax.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
            except Exception:
                pass
    ax.set_xticks(list(x))
    ax.set_xticklabels(phases, rotation=20, fontsize=9)
    ax.set_ylabel('Runs per Over (RPO)')
    ax.set_xlabel('Match Phase')
    ax.set_title('Phase-wise RPO per Team')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    # ensure legend shows with a frame to improve readability
    try:
        leg = ax.legend(title='Team', fontsize=9)
        leg.get_frame().set_alpha(0.9)
    except Exception:
        pass
    fig.tight_layout(pad=1.0)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # save with transparent background False for better visibility in markdown viewers
        fig.savefig(out_path, format='png', bbox_inches='tight')
        plt.close(fig)
        # return a path suitable for Markdown linking relative to workspace
        return str(out_path.as_posix())
    # If the caller wants to embed interactively, return the figure before closing it
    if hasattr(plt, 'Figure'):
        return fig
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        return f"data:image/png;base64,{b64}"


def monte_carlo_simulation(deliveries: List[Dict[str, Any]], raw_json: Dict[str, Any], trials: int = 1000) -> Dict[str, Any]:
    """Monte Carlo simulation using empirical over totals (bootstrap sampling of observed overs).
    Returns summary statistics (win probabilities and sample score distribution percentiles).
    """
    match_id = derive_match_id_from_csv(deliveries, raw_json)
    # collect observed over totals per innings from deliveries (prefer match-scoped)
    over_totals_by_inn: Dict[int, List[int]] = {}
    for d in deliveries:
        if match_id is not None and d.get('match_id') != match_id:
            continue
        inn = d.get('innings') or 1
        ov = d.get('over')
        if ov is None:
            continue
        key = (inn, ov)
        over_totals_by_inn.setdefault(inn, [])
    # compute totals
    # build mapping inn -> over -> total
    inn_over_totals: Dict[int, Dict[int, int]] = {}
    for d in deliveries:
        if match_id is not None and d.get('match_id') != match_id:
            continue
        inn = d.get('innings') or 1
        ov = d.get('over')
        if ov is None:
            continue
        inn_over_totals.setdefault(inn, {})
        inn_over_totals[inn][ov] = inn_over_totals[inn].get(ov, 0) + (d.get('runs_off_bat') or 0) + (d.get('extras') or 0)

    # fallback: if no over data, return empty
    if not inn_over_totals:
        return {'trials': 0, 'note': 'No over totals available for simulation.'}

    # number of overs per innings: infer from max over in that innings
    overs_per_inn = {inn: (max(overs.keys()) + 1 if overs else 0) for inn, overs in inn_over_totals.items()}

    # turn over totals dict into list per innings for bootstrap sampling
    samples_per_inn: Dict[int, List[int]] = {inn: list(overs.values()) for inn, overs in inn_over_totals.items()}

    import statistics as _stats
    results = []
    for _ in range(trials):
        scores = {}
        for inn, over_list in samples_per_inn.items():
            n_overs = overs_per_inn.get(inn, len(over_list))
            if not over_list or n_overs <= 0:
                scores[inn] = 0
            else:
                # sample n_overs with replacement and sum
                s = sum(random.choice(over_list) for _ in range(n_overs))
                scores[inn] = s
        results.append(scores)

    # compute simple win probabilities comparing innings 1 vs 2 when both present
    win_a = win_b = ties = 0
    totals_inn1 = [r.get(1, 0) for r in results]
    totals_inn2 = [r.get(2, 0) for r in results]
    for a, b in zip(totals_inn1, totals_inn2):
        if a > b:
            win_a += 1
        elif b > a:
            win_b += 1
        else:
            ties += 1

    trials_eff = len(totals_inn1)
    summary = {
        'trials': trials_eff,
        'win_prob_innings1': round(win_a / trials_eff, 3) if trials_eff else 0.0,
        'win_prob_innings2': round(win_b / trials_eff, 3) if trials_eff else 0.0,
        'tie_prob': round(ties / trials_eff, 3) if trials_eff else 0.0,
        'inn1_stats': {
            'median': _stats.median(totals_inn1) if totals_inn1 else 0,
            'mean': round(_stats.mean(totals_inn1), 1) if totals_inn1 else 0,
        },
        'inn2_stats': {
            'median': _stats.median(totals_inn2) if totals_inn2 else 0,
            'mean': round(_stats.mean(totals_inn2), 1) if totals_inn2 else 0,
        }
    }
    return summary


if __name__ == "__main__":
    import argparse
    import csv

    ap = argparse.ArgumentParser(description="Analyze a cricket match JSON or CSVs and output Markdown insights.")
    ap.add_argument("json", type=str, nargs='?', help="Optional: Path to match JSON (e.g., 1244025.json). If omitted, CSVs will be used when possible.")
    ap.add_argument("--match-id", type=int, default=None, help="Optional: match_id to scope analysis from CSV files directly (skips JSON).")
    ap.add_argument("--out", type=str, default=None, help="Output Markdown path")
    ap.add_argument("--plots", action="store_true", help="Generate plots and embed base64 images in the Markdown if matplotlib is available")
    ap.add_argument("--monte", type=int, default=0, help="Run Monte Carlo simulation with given number of trials (e.g., --monte 1000)")
    args = ap.parse_args()

    # Determine a filename stem early so plots can be written to disk with a predictable name
    if args.out:
        try:
            stem = Path(args.out).stem
        except Exception:
            stem = None
    elif args.json:
        stem = Path(args.json).stem
    elif args.match_id is not None:
        stem = f"match_{args.match_id}"
    else:
        stem = None

    # Prepare images directory under insights/
    images_dir = Path(__file__).resolve().parents[1] / "insights" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Locate deliveries.csv (prefer JSON-side folder if a JSON path is given)
    deliveries = None
    candidates = [Path("deliveries.csv")]
    if args.json:
        try:
            candidates.insert(0, Path(args.json).resolve().parent / "deliveries.csv")
        except Exception:
            pass
    for c in candidates:
        if c.exists():
            try:
                deliveries = load_deliveries_csv(c)
            except Exception:
                deliveries = None
            break

    ins = None
    raw_data: Dict[str, Any] = {'info': {}}

    if args.match_id is not None:
        # CSV-only mode scoped by match_id
        if deliveries is None:
            print("No deliveries.csv found for CSV-only analysis. Place deliveries.csv in the workspace or provide a JSON file.")
            sys.exit(1)
        # Try to populate teams/dates from matches.csv
        matches_path = Path('matches.csv')
        if matches_path.exists():
            with open(matches_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    mid = safe_int(r.get('match_id'))
                    if mid is None:
                        continue
                    if mid == args.match_id:
                        raw_data['info']['teams'] = [r.get('team_a'), r.get('team_b')]
                        raw_data['info']['dates'] = [r.get('date')]
                        break
        # Build Insights minimal object
        ins = Insights(match=parse_match_summary(raw_data if raw_data.get('info') else {'info': {}}), innings=[], key_points=[], tactical_notes=[])
        # Filter deliveries
        deliveries_filtered = [d for d in deliveries if d.get('match_id') == args.match_id]
        ins.csv_summary = {}
        ins.csv_summary['phase_run_rates'] = compute_phase_run_rates(deliveries_filtered, raw_data)
        ins.csv_summary['phase_run_rates_per_team'] = compute_phase_run_rates_per_team(deliveries_filtered, raw_data)
        ins.csv_summary['wicket_clusters'] = compute_wicket_clusters(deliveries_filtered, raw_data)
        ins.csv_summary['bowler_wicket_clusters'] = compute_per_bowler_wicket_clusters(deliveries_filtered, raw_data)

    elif args.json:
        # Use provided JSON as primary source, with optional CSV enrichment
        ins = analyze_match(Path(args.json))
        try:
            raw_data = json.loads(Path(args.json).read_text(encoding='utf-8'))
        except Exception:
            raw_data = {'info': {}}

    else:
        # No JSON and no explicit match-id: try to use deliveries.csv and infer the match
        if deliveries is None:
            print("No input JSON provided and no deliveries.csv found. Nothing to analyze.")
            sys.exit(1)
        inferred_mid = derive_match_id_from_csv(deliveries, {'info': {}})
        deliveries_filtered = [d for d in deliveries if d.get('match_id') == inferred_mid] if inferred_mid is not None else deliveries
        # Try to populate teams/dates from matches.csv
        matches_path = Path('matches.csv')
        if matches_path.exists() and inferred_mid is not None:
            with open(matches_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    mid = safe_int(r.get('match_id'))
                    if mid is None:
                        continue
                    if mid == inferred_mid:
                        raw_data['info']['teams'] = [r.get('team_a'), r.get('team_b')]
                        raw_data['info']['dates'] = [r.get('date')]
                        break
        ins = Insights(match=parse_match_summary(raw_data if raw_data.get('info') else {'info': {}}), innings=[], key_points=[], tactical_notes=[])
        ins.csv_summary = {}
        ins.csv_summary['phase_run_rates'] = compute_phase_run_rates(deliveries_filtered, raw_data)
        ins.csv_summary['phase_run_rates_per_team'] = compute_phase_run_rates_per_team(deliveries_filtered, raw_data)
        ins.csv_summary['wicket_clusters'] = compute_wicket_clusters(deliveries_filtered, raw_data)
        ins.csv_summary['bowler_wicket_clusters'] = compute_per_bowler_wicket_clusters(deliveries_filtered, raw_data)

    # Further enrichment if we have both JSON-based raw_data and CSV deliveries
    if deliveries is not None and ins is not None:
        try:
            deliveries_for_use = deliveries
            # If we filtered earlier, prefer that
            df = locals().get('deliveries_filtered', None)
            if df is not None:
                deliveries_for_use = df
            ins.csv_summary = ins.csv_summary or {}
            ins.csv_summary['phase_run_rates'] = compute_phase_run_rates(deliveries_for_use, raw_data)
            ins.csv_summary['phase_run_rates_per_team'] = compute_phase_run_rates_per_team(deliveries_for_use, raw_data)
            ins.csv_summary['wicket_clusters'] = compute_wicket_clusters(deliveries_for_use, raw_data)
            ins.csv_summary['bowler_wicket_clusters'] = compute_per_bowler_wicket_clusters(deliveries_for_use, raw_data)
            if args.plots and plt is not None:
                ins.plots = ins.plots or {}
                # Save plot to images_dir using the stem when possible; otherwise fall back to data URI
                out_png = None
                if stem:
                    out_png = images_dir / f"{stem}_phase_rpo_per_team.png"
                img = plot_phase_run_rates_per_team(ins.csv_summary.get('phase_run_rates_per_team', {}), out_path=out_png)
                if img:
                    # img returns either a data URI or an absolute posix path to the saved PNG
                    if isinstance(img, str) and img.startswith('data:image'):
                        ins.plots['phase_rpo_per_team'] = img
                    else:
                        pname = Path(img).name
                        ins.plots['phase_rpo_per_team'] = f"insights/images/{pname}"
        except Exception:
            pass

    if args.monte and deliveries is not None and ins is not None:
        try:
            deliveries_for_monte = [d for d in deliveries if (args.match_id is None or d.get('match_id') == args.match_id)]
            ins.monte = monte_carlo_simulation(deliveries_for_monte, raw_data, trials=args.monte)
        except Exception:
            ins.monte = {'error': 'monte simulation failed'}

    # Render and write output
    md = insights_to_markdown(ins)
    if args.out:
        out = Path(args.out)
    else:
        if args.json:
            stem = Path(args.json).stem
        elif args.match_id is not None:
            stem = f"match_{args.match_id}"
        else:
            inferred_mid = derive_match_id_from_csv(deliveries, {'info': {}}) if deliveries is not None else None
            stem = f"match_{inferred_mid}" if inferred_mid is not None else "insights_output"
        out = Path(__file__).resolve().parents[1] / "insights" / (stem + "_insights.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote insights to {out}")
