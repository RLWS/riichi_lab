"""Read-only tensor identity, independent of optimizer/metadata packaging."""

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    import torch

    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    result = {}
    for task in ("all", "turn", "response"):
        digest = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            if task == "turn" and name.startswith("response_"):
                continue
            if task == "response" and name.startswith("turn_"):
                continue
            digest.update(
                (
                    name
                    + "|"
                    + str(tensor.dtype)
                    + "|"
                    + str(list(tensor.shape))
                    + "\n"
                ).encode()
            )
            digest.update(
                tensor.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        result[task] = digest.hexdigest()
    return {
        "checkpoint": str(path),
        "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "tensor_fingerprints": result,
        "algorithm": "sorted parameter names + dtype + shape + contiguous raw bytes; excludes optimizer/config/summary",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError("output must be new")
    result = [fingerprint(path) for path in args.checkpoint]
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
