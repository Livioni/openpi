"""Upload a local checkpoint directory to a Hugging Face model repository."""

import argparse
import getpass
import os
from pathlib import Path

from huggingface_hub import HfApi

INFERENCE_ALLOW_PATTERNS = [
    "params/**",
    "assets/**",
    "**/params/**",
    "**/assets/**",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload only the inference files from a checkpoint directory to a Hugging Face model repository."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Local checkpoint directory, for example checkpoints/pi0_put_mango.",
    )
    parser.add_argument("repo_id", help="Target Hugging Face repository in OWNER/REPO format.")
    parser.add_argument(
        "--token",
        help="Hugging Face write token. If omitted, HF_TOKEN or a hidden prompt is used.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="Optional destination directory inside the repository (default: repository root).",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message (default: Upload checkpoint <directory name>).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target repository as private if it does not exist.",
    )
    return parser.parse_args()


def get_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("HF_TOKEN")
    if token is None:
        token = getpass.getpass("Hugging Face token: ")
    if not token.strip():
        raise ValueError("A non-empty Hugging Face token is required.")
    return token.strip()


def find_inference_checkpoints(checkpoint_dir: Path) -> list[Path]:
    checkpoints = []
    for params_dir in checkpoint_dir.rglob("params"):
        if not params_dir.is_dir() or not any(path.is_file() for path in params_dir.rglob("*")):
            continue

        candidate = params_dir.parent
        assets_dir = candidate / "assets"
        if assets_dir.is_dir() and any(assets_dir.glob("*/norm_stats.json")):
            checkpoints.append(candidate)

    return sorted(checkpoints)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist or is not a directory: {checkpoint_dir}")
    inference_checkpoints = find_inference_checkpoints(checkpoint_dir)
    if not inference_checkpoints:
        raise ValueError(
            "No deployable checkpoint found. Expected a checkpoint containing params/ and "
            f"assets/<asset_id>/norm_stats.json under: {checkpoint_dir}"
        )

    token = get_token(args.token)
    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    relative_checkpoints = [str(path.relative_to(checkpoint_dir)) for path in inference_checkpoints]
    print(f"Uploading inference files from {checkpoint_dir} to {args.repo_id} ...")
    print(f"Deployable checkpoints: {', '.join(relative_checkpoints)}")
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=checkpoint_dir,
        path_in_repo=args.path_in_repo or None,
        commit_message=args.commit_message or f"Upload checkpoint {checkpoint_dir.name}",
        allow_patterns=INFERENCE_ALLOW_PATTERNS,
    )
    print(f"Upload complete: {commit.commit_url or repo_url}")


if __name__ == "__main__":
    main()
