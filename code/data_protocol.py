"""Auditable match-level splits. No training library is needed to audit JSON files."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "prefix-v2"


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ordered_players(match):
    players = match.get("players") or []
    slots = [p.get("player_slot") for p in players]
    if len(players) != 10 or any(not isinstance(s, int) for s in slots) or len(set(slots)) != 10:
        raise ValueError("Expected ten distinct player slots")
    players = sorted(players, key=lambda p: p["player_slot"])
    if sum(p["player_slot"] < 128 for p in players) != 5:
        raise ValueError("Expected five players on each side")
    return players


def inspect_file(path):
    row = {"file": path.name, "bytes": path.stat().st_size, "errors": [], "eligible": {}}
    try:
        raw = path.read_bytes()
        row["sha256"] = hashlib.sha256(raw).hexdigest()
        d = json.loads(raw)
        if not isinstance(d, dict):
            raise ValueError("JSON is not a match object")
        for key in ("match_id", "duration", "start_time", "patch", "leagueid", "lobby_type", "game_mode", "version", "radiant_win"):
            row[key] = d.get(key)
        if not isinstance(d.get("match_id"), int):
            row["errors"].append("missing_match_id")
        if not isinstance(d.get("radiant_win"), bool):
            row["errors"].append("missing_outcome")
        if not finite_number(d.get("duration")) or d["duration"] <= 0:
            row["errors"].append("invalid_duration")
        if not finite_number(d.get("start_time")) or d["start_time"] <= 0:
            row["errors"].append("invalid_start_time")
        players = ordered_players(d)
        if any(not isinstance(p.get("hero_id"), int) or p["hero_id"] <= 0 for p in players):
            row["errors"].append("missing_hero")
        for p in players:
            times = p.get("times") or []
            if not times or any(not finite_number(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
                row["errors"].append("invalid_player_times")
                continue
            for key in ("gold_t", "xp_t"):
                values = p.get(key) or []
                if len(values) != len(times) or any(not finite_number(v) for v in values):
                    row["errors"].append("invalid_" + key)
        row["errors"] = sorted(set(row["errors"]))
        # Use actual recorded timestamps, not array positions, as the availability contract.
        available = set(players[0].get("times") or [])
        for p in players[1:]:
            available.intersection_update(p.get("times") or [])
        max_minute = 0
        while (max_minute + 1) * 60 in available and max_minute < 240:
            max_minute += 1
        row["max_prefix_minute"] = max_minute if 0 in available else -1
        row["max_hero_id"] = max(p.get("hero_id", 0) or 0 for p in players)
        row["has_leaver"] = any((p.get("leaver_status") or 0) > 1 for p in players)
        row["has_final_actions"] = any(p.get("actions_per_min") is not None for p in players)
        for t in (10, 20, 30, 40):
            row["eligible"][str(t)] = not row["errors"] and d["duration"] > t * 60 and row["max_prefix_minute"] >= t
        row["legacy_ended_by_minute"] = {str(t): bool(finite_number(d.get("duration")) and d["duration"] <= t * 60) for t in (10, 20, 30, 40)}
    except (ValueError, TypeError, KeyError, OSError) as exc:
        row["errors"].append(type(exc).__name__ + ": " + str(exc))
    return row


def chronological_split(rows, train_fraction=0.7, val_fraction=0.15):
    """Never split a start-time tie across partitions; duplicate IDs must be removed first."""
    rows = sorted(rows, key=lambda r: (r["start_time"], r["match_id"]))
    if len(rows) < 3:
        raise ValueError("At least three valid matches are required")
    a = max(1, int(len(rows) * train_fraction))
    b = max(a + 1, int(len(rows) * (train_fraction + val_fraction)))
    while a < len(rows) and rows[a]["start_time"] == rows[a - 1]["start_time"]:
        a += 1
    b = max(b, a + 1)
    while b < len(rows) and rows[b]["start_time"] == rows[b - 1]["start_time"]:
        b += 1
    result = {"train": rows[:a], "validation": rows[a:b], "test": rows[b:]}
    if any(not group for group in result.values()):
        raise ValueError("Not enough distinct timestamps for three chronological partitions")
    return result


def eligible_rows(manifest, split, minute):
    return [r for r in manifest["splits"][split] if r["duration"] > minute * 60 and r["max_prefix_minute"] >= minute]


def load_manifest(path):
    manifest = read_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema; regenerate the audit")
    if manifest["fingerprint"] != fingerprint(manifest["splits"]):
        raise ValueError("Manifest fingerprint mismatch")
    ids = [r["match_id"] for group in manifest["splits"].values() for r in group]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate match IDs across the manifest")
    return manifest


def audit(data_dir, output_dir, workers=4, limit=None):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    files = sorted(data_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(inspect_file, files)):
            rows.append(row)
            if (i + 1) % 2000 == 0:
                print(f"Audited {i + 1}/{len(files)}", flush=True)
    seen, valid = set(), []
    for row in rows:
        mid = row.get("match_id")
        if mid in seen:
            row["errors"].append("duplicate_match_id")
        if not row["errors"]:
            seen.add(mid)
            if row["max_prefix_minute"] >= 5 and row["duration"] > 300:
                valid.append(row)
            else:
                row["errors"].append("no_five_minute_prefix")
    splits = chronological_split(valid)
    manifest = {"schema_version": SCHEMA_VERSION, "data_dir": str(data_dir), "debug_subset": bool(limit),
                "split_method": "chronological_70_15_15_start_time_ties_kept_together",
                "fingerprint": fingerprint(splits), "splits": splits}
    write_json(output_dir / "manifest.json", manifest)
    with (output_dir / "audit_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    legacy = [r for r in rows if r["bytes"] > 100 * 1024 and "duration" in r and finite_number(r["duration"])]
    summary = {
        "files_scanned": len(files), "valid_unique_matches": len(valid), "excluded_files": len(files) - len(valid),
        "debug_subset": bool(limit), "exclusion_reasons": dict(Counter(e for r in rows for e in r["errors"])),
        "legacy_size_filter_count": len(legacy),
        "legacy_ended_match_counts": {str(t): sum(r["duration"] <= t * 60 for r in legacy) for t in (10, 20, 30, 40)},
        "patch_counts": dict(Counter(str(r.get("patch")) for r in valid)),
        "lobby_type_counts": dict(Counter(str(r.get("lobby_type")) for r in valid)),
        "game_mode_counts": dict(Counter(str(r.get("game_mode")) for r in valid)),
        "league_id_counts": dict(Counter(str(r.get("leagueid")) for r in valid)),
        "matches_with_leavers": sum(r["has_leaver"] for r in valid),
        "splits": {},
    }
    for name, group in splits.items():
        summary["splits"][name] = {
            "n": len(group), "radiant_win_rate": sum(r["radiant_win"] for r in group) / len(group),
            "start_utc": datetime.fromtimestamp(group[0]["start_time"], timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(group[-1]["start_time"], timezone.utc).isoformat(),
            "patch_counts": dict(Counter(str(r.get("patch")) for r in group)),
            "eligible_by_minute": {str(t): len(eligible_rows(manifest, name, t)) for t in (10, 20, 30, 40)},
        }
    write_json(output_dir / "audit_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("files_scanned", "valid_unique_matches", "excluded_files", "legacy_ended_match_counts")}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dota2_pro_matches")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "data_audit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Debug audit only; never a publication dataset")
    args = parser.parse_args()
    audit(args.data_dir, args.output_dir, args.workers, args.limit)
