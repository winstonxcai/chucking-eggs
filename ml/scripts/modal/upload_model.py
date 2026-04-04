"""Upload/download model checkpoints to/from Modal volume.

Usage:
    python scripts/modal/upload_model.py download --remote model_final.pt --local ./model_final.pt
    python scripts/modal/upload_model.py upload --local ./model_final.pt --remote model_final.pt
    python scripts/modal/upload_model.py list
"""

from __future__ import annotations

import argparse

import modal

vol = modal.Volume.from_name("guandan-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"

app = modal.App("guandan-model-utils")


@app.function(volumes={CHECKPOINT_DIR: vol})
def list_checkpoints() -> list[str]:
    import os

    files = []
    for f in os.listdir(CHECKPOINT_DIR):
        path = os.path.join(CHECKPOINT_DIR, f)
        size = os.path.getsize(path)
        files.append(f"{f} ({size / 1024:.1f} KB)")
    return files


@app.function(volumes={CHECKPOINT_DIR: vol})
def read_checkpoint(remote_name: str) -> bytes:
    import os

    path = os.path.join(CHECKPOINT_DIR, remote_name)
    with open(path, "rb") as f:
        return f.read()


@app.function(volumes={CHECKPOINT_DIR: vol})
def write_checkpoint(remote_name: str, data: bytes) -> str:
    import os

    path = os.path.join(CHECKPOINT_DIR, remote_name)
    with open(path, "wb") as f:
        f.write(data)
    vol.commit()
    return f"Uploaded {len(data)} bytes to {remote_name}"


@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(description="Manage Modal checkpoints")
    sub = parser.add_subparsers(dest="command")

    ls = sub.add_parser("list", help="List checkpoints on Modal")

    dl = sub.add_parser("download", help="Download checkpoint from Modal")
    dl.add_argument("--remote", required=True, help="Remote filename")
    dl.add_argument("--local", required=True, help="Local path to save")

    ul = sub.add_parser("upload", help="Upload checkpoint to Modal")
    ul.add_argument("--local", required=True, help="Local file to upload")
    ul.add_argument("--remote", required=True, help="Remote filename")

    args = parser.parse_args()

    if args.command == "list":
        files = list_checkpoints.remote()
        print("Checkpoints on Modal:")
        for f in files:
            print(f"  {f}")

    elif args.command == "download":
        data = read_checkpoint.remote(args.remote)
        with open(args.local, "wb") as f:
            f.write(data)
        print(f"Downloaded {args.remote} → {args.local} ({len(data)} bytes)")

    elif args.command == "upload":
        with open(args.local, "rb") as f:
            data = f.read()
        result = write_checkpoint.remote(args.remote, data)
        print(result)

    else:
        parser.print_help()
