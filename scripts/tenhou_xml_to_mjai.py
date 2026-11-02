"""Export a masked MJAI stream using the pinned development parser."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seat", required=True, type=int, choices=range(4))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.append(str(repo / ".venv/Lib/site-packages"))
    from mjai.data.tenhou.parser import TenhouParser
    from rilab.parity import convert_event

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"type": "start_game", "id": args.seat}) + "\n")
        started = False
        for event in TenhouParser().parse_file(args.input):
            if event.event_type == "go" and event.payload["type"] & 16:
                raise ValueError("sanma is unsupported")
            if event.event_type == "init":
                if started:
                    log.write(json.dumps({"type": "end_kyoku"}) + "\n")
                started = True
            converted = convert_event(event, args.seat)
            if converted:
                log.write(json.dumps(converted, ensure_ascii=False) + "\n")
        if started:
            log.write(json.dumps({"type": "end_kyoku"}) + "\n")
        log.write(json.dumps({"type": "end_game"}) + "\n")


if __name__ == "__main__":
    main()
