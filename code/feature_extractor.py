"""Prefix-only telemetry. Minute t is the snapshot at exactly 60*t seconds.

Purchases are acquisitions, not current inventory. Resource rank is a proxy,
not a validated Position 1--5 label. Graphs use timestamped kills; retrospective
teamfight summaries, lane assignments and final-match summaries are unused.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import numpy as np

from data_protocol import ROOT, SCHEMA_VERSION, ordered_players, read_json, fingerprint

NODE_FEATURES = [
    "radiant", "cumulative_gold_10000", "cumulative_xp_10000", "wards_observer_20",
    "wards_sentry_20", "runes_10", "kills_20", "last_hits_500", "denies_100",
    "distinct_purchases_30", "purchase_prior_early", "purchase_prior_mid", "purchase_prior_late",
    "resource_rank_weight", "minute_60", "radiant_towers_destroyed_11", "dire_towers_destroyed_11",
    "roshan_events_3", "five_minute_gold_change_10000", "five_minute_xp_change_10000",
]
TS_FEATURES = ["gold_difference_10000", "xp_difference_10000", "gold_first_difference_10000", "xp_first_difference_10000"]
ROLE_COLUMN = NODE_FEATURES.index("resource_rank_weight")
EDGE_FEATURES = ["kills_last_5_minutes", "kills_since_start"]


@lru_cache(maxsize=1)
def hero_metadata():
    return read_json(ROOT / "hero_metadata.json")


def timestamped(log, cutoff, lower=None):
    return [event for event in (log or []) if isinstance(event.get("time"), (int, float))
            and event["time"] <= cutoff and (lower is None or event["time"] > lower)]


def value_at(player, key, second, required=True):
    times, values = player.get("times") or [], player.get(key) or []
    try:
        index = times.index(second)
        value = values[index]
        if not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError("Non-finite telemetry")
        return float(value)
    except (ValueError, IndexError):
        if required:
            raise ValueError(f"Missing {key} snapshot at {second}s")
        return 0.0


def team_differences(players, second):
    return np.array([sum((1 if i < 5 else -1) * value_at(p, key, second) for i, p in enumerate(players))
                     for key in ("gold_t", "xp_t")], dtype=np.float32)


def fit_purchase_prior(rows, data_dir, manifest_fingerprint):
    """Train-fold-only acquisition frequency; no outcomes or external win statistics."""
    counts = defaultdict(lambda: [Counter(), Counter(), Counter()])
    for i, row in enumerate(rows):
        d = read_json(data_dir / row["file"])
        for p in ordered_players(d):
            seen = [set(), set(), set()]
            for e in timestamped(p.get("purchase_log"), min(d["duration"], 1800)):
                bucket = 0 if e["time"] <= 720 else (1 if e["time"] <= 1200 else 2)
                if isinstance(e.get("key"), str):
                    seen[bucket].add(e["key"].removeprefix("item_"))
            for bucket in range(3):
                counts[str(p["hero_id"])][bucket].update(seen[bucket])
        if (i + 1) % 2000 == 0:
            print(f"Fitted prior on {i + 1}/{len(rows)} training matches", flush=True)
    items = {hero: [[k for k, _ in sorted(c.items(), key=lambda pair: (-pair[1], pair[0]))[:3]] for c in stages]
             for hero, stages in counts.items()}
    return {"schema_version": SCHEMA_VERSION, "manifest_fingerprint": manifest_fingerprint,
            "training_ids_hash": fingerprint([r["match_id"] for r in rows]),
            "method": "top3_acquisition_frequency_per_hero_per_phase_training_only", "items": items}


class MatchFeatureExtractor:
    def __init__(self, match_data, target_minute, prior=None, aux_horizon=5):
        if not isinstance(target_minute, int) or target_minute < 0:
            raise ValueError("target_minute must be a nonnegative integer")
        self.data, self.target_minute = match_data, target_minute
        self.target_sec = target_minute * 60
        self.players = ordered_players(match_data)
        if match_data.get("duration", 0) <= self.target_sec:
            raise ValueError("Match already ended at the requested prediction time")
        self.prior = (prior or {}).get("items", {})
        self.aux_horizon = aux_horizon

    def extract_inputs(self):
        players, cutoff, t = self.players, self.target_sec, self.target_minute
        # Require actual snapshots: never clip to final values or forward-fill after a match.
        differences = np.stack([team_differences(players, minute * 60) for minute in range(t + 1)]) / 10000
        changes = np.diff(differences, axis=0, prepend=differences[:1])
        sequence = np.concatenate((differences, changes), axis=1).astype(np.float32)
        roles = np.zeros(10, dtype=np.float32)
        for start in (0, 5):
            ranks = sorted(range(start, start + 5), key=lambda i: (-value_at(players[i], "gold_t", cutoff), players[i]["player_slot"]))
            for rank, i in enumerate(ranks, start=1):
                roles[i] = (6 - rank) / 5
        rad_towers = dire_towers = roshan = 0
        for event in timestamped(self.data.get("objectives"), cutoff):
            key = str(event.get("key", ""))
            if event.get("type") == "BUILDING_KILL" and "tower" in key:
                rad_towers += int("badguys" in key)
                dire_towers += int("goodguys" in key)
            roshan += int(event.get("type") == "CHAT_MESSAGE_ROSHAN_KILL")
        momentum = differences[-1] - differences[max(0, t - 5)]
        nodes = []
        for i, p in enumerate(players):
            purchases = {e["key"].removeprefix("item_") for e in timestamped(p.get("purchase_log"), cutoff) if isinstance(e.get("key"), str)}
            stages = self.prior.get(str(p["hero_id"]), [[], [], []])
            rates = [len(purchases.intersection(items)) / len(items) if items else 0.0 for items in stages]
            nodes.append([
                float(i < 5), value_at(p, "gold_t", cutoff) / 10000, value_at(p, "xp_t", cutoff) / 10000,
                len(timestamped(p.get("obs_log"), cutoff)) / 20, len(timestamped(p.get("sen_log"), cutoff)) / 20,
                len(timestamped(p.get("runes_log"), cutoff)) / 10, len(timestamped(p.get("kills_log"), cutoff)) / 20,
                value_at(p, "lh_t", cutoff, required=False) / 500, value_at(p, "dn_t", cutoff, required=False) / 100,
                len(purchases) / 30, *rates, roles[i], t / 60,
                rad_towers / 11, dire_towers / 11, roshan / 3, *momentum,
            ])
        # Dense 10-node representation: [target i, source j, edge channel].
        edges = np.zeros((10, 10, 2), dtype=np.float32)
        metadata, name_to_index = hero_metadata(), {}
        for i, p in enumerate(players):
            meta = metadata.get(str(p["hero_id"]))
            if not meta:
                raise ValueError(f"Hero ID {p['hero_id']} missing from frozen metadata")
            name_to_index[meta["name"]] = i
        for source, p in enumerate(players):
            for event in timestamped(p.get("kills_log"), cutoff):
                target = name_to_index.get(event.get("key"))
                if target is not None and source != target:
                    edges[target, source, 1] += 1
                    edges[target, source, 0] += int(event["time"] > cutoff - 300)
        return {"nodes": np.asarray(nodes, dtype=np.float32),
                "heroes": np.array([p["hero_id"] for p in players], dtype=np.int64),
                "edges": np.log1p(edges), "roles": roles, "sequence": sequence}

    def extract_targets(self):
        if not isinstance(self.data.get("radiant_win"), bool):
            raise ValueError("A binary match outcome is required")
        future = self.target_sec + self.aux_horizon * 60
        valid, delta = self.data["duration"] > future, np.zeros(2, dtype=np.float32)
        if valid:
            try:
                delta = (team_differences(self.players, future) - team_differences(self.players, self.target_sec)) / 10000
            except ValueError:
                valid = False
        # Missing future labels mask only auxiliary loss, never current-time eligibility.
        return {"outcome": np.float32(self.data["radiant_win"]), "aux": delta,
                "aux_mask": np.float32(valid), "match_id": np.int64(self.data["match_id"])}

    def extract(self):
        return {**self.extract_inputs(), **self.extract_targets()}


def feature_contract():
    return {"schema_version": SCHEMA_VERSION, "node_features": NODE_FEATURES, "time_features": TS_FEATURES,
            "edge_features": EDGE_FEATURES, "graph": "directed_timestamped_kills",
            "roles": "current_cumulative_gold_rank_proxy_not_lane_positions",
            "hero_metadata_sha256": hashlib.sha256((ROOT / "hero_metadata.json").read_bytes()).hexdigest()}
