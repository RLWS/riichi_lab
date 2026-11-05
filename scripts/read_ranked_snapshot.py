"""Fetch a sanitized read-only cloud snapshot using an interactive SSH password."""

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shlex


def main():
    import paramiko

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote-root", default="/usr/local/rlws/riichi_lab/outputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("snapshot output must be new")
    source = (
        Path(__file__).resolve().parents[1] / "rilab/ranked_snapshot.py"
    ).read_text(encoding="utf-8")
    command = (
        "nice -n 15 python3 -c "
        + shlex.quote(
            source
            + "\nimport sys; print(json.dumps(snapshot(sys.argv[1]), ensure_ascii=False))"
        )
        + " "
        + shlex.quote(args.remote_root)
    )
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(Path.home() / ".ssh/known_hosts"))
    try:
        password = getpass.getpass("SSH password (not saved): ")
        client.connect(
            args.host,
            port=args.port,
            username=args.user,
            password=password,
            allow_agent=False,
            look_for_keys=False,
            timeout=15,
            auth_timeout=20,
        )
        del password
        client.get_transport().set_keepalive(15)
        _, stdout, stderr = client.exec_command(command, timeout=600)
        data = stdout.read()
        error = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(
                f"read-only extraction failed ({status}): {error[:1000]}"
            )
        result = json.loads(data)
        result["extractor_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "saved": str(args.output),
                    "batches": len(result["batches"]),
                    "sessions": sum(len(b["games"]) for b in result["batches"]),
                    "read_errors": len(result["read_errors"]),
                }
            )
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
