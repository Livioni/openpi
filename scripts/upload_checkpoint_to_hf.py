"""Upload a local checkpoint directory to a Hugging Face model repository."""

import argparse
import getpass
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the contents of a checkpoint directory to a Hugging Face model repository."
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


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory does not exist or is not a directory: {checkpoint_dir}")
    if not any(checkpoint_dir.iterdir()):
        raise ValueError(f"Checkpoint directory is empty: {checkpoint_dir}")

    token = get_token(args.token)
    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading {checkpoint_dir} to {args.repo_id} ...")
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=checkpoint_dir,
        path_in_repo=args.path_in_repo or None,
        commit_message=args.commit_message or f"Upload checkpoint {checkpoint_dir.name}",
    )
    print(f"Upload complete: {commit.commit_url or repo_url}")


if __name__ == "__main__":
    main()
