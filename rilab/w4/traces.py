"""Read archived ranked traces and export only perspective-state and decision fields."""

import argparse
import getpass
import gzip
import hashlib
import json
import re
import shlex
import tarfile
from collections import Counter
from pathlib import Path

from .public import CONFIG, read, save

STATE = """actor hand dealer round_wind round_number match_rounds seat_wind scores honba
riichi_sticks remaining_tiles turn_number all_discards discard_history all_melds own_melds
dora_indicators public_events riichi_declared riichi_declared_by_player riichi_turn_by_player
temporary_furiten riichi_furiten last_discard_actor last_discard_tile feature_schema
own_discards""".split()
PROFILES = {"full_state_v2": 493, "full_state_v3": 521}


def fields(row, keys):
    return {k: row[k] for k in keys if k in row}


def export(stream, directory, expected=None):
    digest = hashlib.sha256()
    requests = {}
    hands = []
    accepted = set()
    hand = None
    seat = scores = None
    anomalies = Counter()
    for line in stream:
        digest.update(line)
        try:
            r = json.loads(line)
        except ValueError:
            anomalies["invalid_line"] += 1
            continue
        kind = r.get("kind")
        if kind == "received":
            e = r["message"]
            action = e.get("type")
            actor = e.get("actor")
            if action == "start_game":
                seat = e["id"]
            elif action == "end_game":
                scores = e["scores"]
            elif action == "start_kyoku":
                accepted.clear()
                hand = dict(open=False, win=False, deal=False, income=0, loss=0)
            elif hand is not None:
                if action == "reach_accepted":
                    accepted.add(actor)
                elif action in ("chi", "pon", "daiminkan") and actor == seat:
                    hand["open"] = True
                elif action == "hora":
                    if actor == seat:
                        hand["win"] = True
                        hand["income"] += e["deltas"][seat]
                    elif e["target"] == seat:
                        hand["deal"] = True
                        hand["loss"] -= e["deltas"][seat]
                elif action == "end_kyoku":
                    hands.append(hand)
                    hand = None
            continue
        rid = r.get("request_id")
        if kind == "prepared":
            if rid in requests:
                raise ValueError("duplicate prepared request")
            row = fields(
                r,
                [
                    "request_id",
                    "task",
                    "hand_id",
                    "candidates",
                    "bindings",
                    "forced_riichi_discard",
                ],
            )
            row.update(
                state=fields(r["state"], STATE),
                hand_index=len(hands),
                opponent_accepted=bool(accepted - {seat}),
            )
            requests[rid] = row
        elif kind in ("prediction", "execution", "reply"):
            if rid not in requests:
                anomalies["unprepared_" + kind] += 1
                continue
            if kind == "prediction":
                result = r["result"]
                data = fields(
                    result,
                    [
                        "task",
                        "selected_action",
                        "predictions",
                        "raw_selected_action",
                        "action_biases",
                        "pass_bias",
                        "no_yaku_call_penalty",
                        "selection_mode",
                    ],
                )
                data["model"] = fields(
                    result.get("model", {}),
                    ["feature_schema", "task", "checkpoint_task"],
                )
            else:
                data = fields(r, ["status", "actual_action", "fallback", "action"])
            requests[rid][kind] = data
    sha = digest.hexdigest()
    if expected and sha != expected:
        raise ValueError("historical trace checksum changed")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (sha + ".jsonl.gz")
    temp = path.with_suffix(".tmp")
    with gzip.open(temp, "wt", encoding="utf8", compresslevel=3) as out:
        for row in requests.values():
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    temp.replace(path)
    return dict(
        trace_sha256=sha,
        seat=seat,
        scores=scores,
        hand_results=hands,
        requests=len(requests),
        fallbacks=sum(
            bool(r.get("reply", {}).get("fallback")) for r in requests.values()
        ),
        anomalies=anomalies,
    )


def local(cfg, root):
    rows = []
    for manifest in Path("outputs").rglob("manifest.json"):
        paths = sorted(manifest.parent.glob("game_*/session.jsonl"))
        if not paths:
            continue
        m = read(manifest)
        backend = m.get("config", {}).get("backend", {})
        if m.get("mode") != "ranked" or backend.get("adapter_profile") not in PROFILES:
            continue
        for p in paths:
            before = p.stat()
            with p.open("rb") as stream:
                row = export(stream, root / "decisions")
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("local ranked session changed during read")
            row.update(
                bot=PROFILES[backend["adapter_profile"]],
                source_id=p.parent.as_posix(),
                trace_mtime_utc=before.st_mtime,
                model_identity=backend.get("model_identity"),
            )
            rows.append(row)
    save(root / "local_sessions.json", rows)


def cloud(cfg, root, host, port, user):
    import paramiko
    from paramiko.ssh_exception import AuthenticationException

    batches = read(cfg["snapshot"])["batches"]
    lookup = {
        g["source_id"] + "/session.jsonl": g
        for b in batches
        if b["backend"]["adapter_profile"] in PROFILES
        for g in b["games"]
    }
    if not all(
        re.fullmatch(r"ranked_[\w.-]+/game_\d+/session\.jsonl", name) for name in lookup
    ):
        raise ValueError("unexpected path in ranked snapshot")
    index = root / "cloud_traces.json"
    rows = read(index) if index.exists() else {}
    names = [
        n
        for n, g in lookup.items()
        if g["trace_sha256"] not in rows
        or not (root / "decisions" / (g["trace_sha256"] + ".jsonl.gz")).exists()
    ]
    if not names:
        return
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home() / ".ssh" / "known_hosts"))
    try:
        client.connect(host, port=port, username=user, timeout=12)
    except AuthenticationException:
        client.connect(
            host,
            port=port,
            username=user,
            password=getpass.getpass(),
            allow_agent=False,
            look_for_keys=False,
            timeout=12,
        )
    try:
        for start in range(0, len(names), 50):
            chunk = names[start : start + 50]
            command = "tar -C " + shlex.quote(cfg["cloud_root"]) + " -czf - -T -"
            stdin, stdout, stderr = client.exec_command(command)
            stdin.write("\n".join(chunk) + "\n")
            stdin.flush()
            stdin.channel.shutdown_write()
            seen = set()
            with tarfile.open(
                fileobj=stdout, mode="r|gz", bufsize=128 * 1024
            ) as archive:
                for member in archive:
                    if member.name not in chunk or not member.isfile():
                        raise ValueError("unexpected archive member")
                    sha = lookup[member.name]["trace_sha256"]
                    rows[sha] = export(
                        archive.extractfile(member), root / "decisions", sha
                    )
                    seen.add(member.name)
            if stdout.channel.recv_exit_status() or seen != set(chunk):
                raise RuntimeError("incomplete remote read")
            save(index, rows)
            print("verified", len(rows), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    args = parser.parse_args()
    cfg = read(CONFIG)
    root = Path(cfg["output"])
    if args.host:
        cloud(cfg, root, args.host, args.port, args.user)
    else:
        local(cfg, root)
