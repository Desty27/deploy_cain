from pathlib import Path
import random
import csv


def main():
    out = Path(__file__).parents[1] / "data" / "demo_players.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    roles = ["batter", "bowler", "allrounder", "keeper"]
    regions = ["North", "South", "East", "West"]
    rows = []
    rng = random.Random(42)
    for i in range(60):
        role = rng.choice(roles)
        row = {
            "player_id": f"P{i+1:03d}",
            "name": f"Player {i+1}",
            "age": rng.randint(18, 34),
            "role": role,
            "league_level": rng.randint(1,5),
            "matches": rng.randint(5, 70),
            "batting_sr": round(rng.uniform(85, 185), 1) if role != "bowler" else "",
            "batting_avg": round(rng.uniform(12, 62), 1) if role != "bowler" else "",
            "bowling_eco": round(rng.uniform(4.5, 10.5), 2) if role != "batter" else "",
            "bowling_avg": round(rng.uniform(12, 55), 1) if role != "batter" else "",
            "fielding_eff": round(rng.uniform(0.6, 0.98), 2),
            "recent_form": round(rng.uniform(0.2, 0.95), 2),
            "region": rng.choice(regions),
        }
        rows.append(row)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
