"""Convert a local LeRobot v3.0 dataset to the v2.1 layout used by openpi."""

from __future__ import annotations

import concurrent.futures
import copy
import fractions
import json
import math
import os
import pathlib
import shutil
import subprocess
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

_V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_CORE_STATS = ("min", "max", "mean", "std", "count")


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False) + "\n")


def _write_jsonlines(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _load_episode_rows(source: pathlib.Path) -> list[dict[str, Any]]:
    episode_files = sorted((source / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No v3 episode metadata found under {source / 'meta' / 'episodes'}")

    table = pa.concat_tables([pq.read_table(path) for path in episode_files], promote_options="default")
    rows = sorted(table.to_pylist(), key=lambda row: row["episode_index"])
    expected_indices = list(range(len(rows)))
    indices = [row["episode_index"] for row in rows]
    if indices != expected_indices:
        raise ValueError(f"Episode indices must be contiguous from zero, got {indices[:10]}...")
    return rows


def _load_tasks(source: pathlib.Path) -> list[dict[str, Any]]:
    tasks_path = source / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing v3 task metadata: {tasks_path}")
    return sorted(pq.read_table(tasks_path).to_pylist(), key=lambda row: row["task_index"])


def _legacy_episode_stats(row: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for feature_name in feature_names:
        feature_stats = {
            stat: row[f"stats/{feature_name}/{stat}"] for stat in _CORE_STATS if f"stats/{feature_name}/{stat}" in row
        }
        if feature_stats:
            stats[feature_name] = feature_stats
    missing = set(feature_names) - set(stats)
    if missing:
        raise ValueError(f"Episode {row['episode_index']} is missing stats for features: {sorted(missing)}")
    return stats


def _legacy_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert v3 feature metadata to the channel-first v2.1 convention."""
    legacy_features = copy.deepcopy(features)
    for feature in legacy_features.values():
        # v2.1 stores the dataset FPS once at the top level.
        feature.pop("fps", None)

        if feature["dtype"] not in {"image", "video"}:
            continue

        names = feature.get("names")
        shape = feature.get("shape")
        if names == ["height", "width", "channels"] and len(shape) == 3:
            height, width, channels = shape
            feature["shape"] = [channels, height, width]
            feature["names"] = ["channels", "height", "width"]

    return legacy_features


def _source_data_path(source: pathlib.Path, info: dict[str, Any], row: dict[str, Any]) -> pathlib.Path:
    relative = info["data_path"].format(
        chunk_index=row["data/chunk_index"],
        file_index=row["data/file_index"],
    )
    return source / relative


def _source_video_path(source: pathlib.Path, info: dict[str, Any], row: dict[str, Any], video_key: str) -> pathlib.Path:
    relative = info["video_path"].format(
        video_key=video_key,
        chunk_index=row[f"videos/{video_key}/chunk_index"],
        file_index=row[f"videos/{video_key}/file_index"],
    )
    return source / relative


def _split_parquet_files(
    source: pathlib.Path,
    staging: pathlib.Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
) -> None:
    rows_by_file: dict[pathlib.Path, list[dict[str, Any]]] = {}
    for row in episode_rows:
        rows_by_file.setdefault(_source_data_path(source, info, row), []).append(row)

    for source_path, rows in tqdm.tqdm(rows_by_file.items(), desc="Splitting parquet files"):
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source parquet file: {source_path}")
        source_table = pq.read_table(source_path)
        file_start_index = min(row["dataset_from_index"] for row in rows)

        for row in rows:
            episode_index = row["episode_index"]
            episode_length = row["length"]
            offset = row["dataset_from_index"] - file_start_index
            episode_table = source_table.slice(offset, episode_length).replace_schema_metadata(None)
            if episode_table.num_rows != episode_length:
                raise ValueError(
                    f"Episode {episode_index} expected {episode_length} rows, got {episode_table.num_rows}"
                )
            if set(episode_table.column("episode_index").to_pylist()) != {episode_index}:
                raise ValueError(f"Parquet slice for episode {episode_index} contains another episode")

            output_path = staging / _V21_DATA_PATH.format(
                episode_chunk=episode_index // info["chunks_size"],
                episode_index=episode_index,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, output_path, compression="zstd")


def _cut_video(job: tuple[pathlib.Path, pathlib.Path, float, float]) -> pathlib.Path:
    source_path, output_path, start, duration = job
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source video: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            format(start, ".17g"),
            "-t",
            format(duration, ".17g"),
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-reset_timestamps",
            "1",
            "-y",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def _split_videos(
    source: pathlib.Path,
    staging: pathlib.Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    video_keys: list[str],
    workers: int,
) -> None:
    jobs = []
    for row in episode_rows:
        episode_index = row["episode_index"]
        for video_key in video_keys:
            start = row[f"videos/{video_key}/from_timestamp"]
            end = row[f"videos/{video_key}/to_timestamp"]
            if end <= start:
                raise ValueError(f"Invalid video timestamps for episode {episode_index}, camera {video_key}")
            output_path = staging / _V21_VIDEO_PATH.format(
                episode_chunk=episode_index // info["chunks_size"],
                video_key=video_key,
                episode_index=episode_index,
            )
            jobs.append((_source_video_path(source, info, row, video_key), output_path, start, end - start))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_cut_video, job) for job in jobs]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Splitting videos"):
            future.result()


def _probe_video(path: pathlib.Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,start_time,duration,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}, got {len(streams)}")
    return streams[0]


def _verify_video(job: tuple[pathlib.Path, int, int]) -> None:
    path, expected_frames, expected_fps = job
    stream = _probe_video(path)
    actual_frames = int(stream["nb_frames"])
    actual_fps = fractions.Fraction(stream["avg_frame_rate"])
    start_time = float(stream.get("start_time", 0.0))
    duration = float(stream["duration"])
    expected_duration = expected_frames / expected_fps

    if actual_frames != expected_frames:
        raise ValueError(f"{path} has {actual_frames} frames; expected {expected_frames}")
    if actual_fps != expected_fps:
        raise ValueError(f"{path} has FPS {actual_fps}; expected {expected_fps}")
    if abs(start_time) > 1 / expected_fps:
        raise ValueError(f"{path} starts at {start_time}s instead of zero")
    if abs(duration - expected_duration) > 1 / expected_fps:
        raise ValueError(f"{path} has duration {duration}s; expected {expected_duration}s")


def _verify_conversion(
    staging: pathlib.Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    video_keys: list[str],
    workers: int,
) -> None:
    data_files = sorted((staging / "data").glob("chunk-*/*.parquet"))
    if len(data_files) != len(episode_rows):
        raise ValueError(f"Expected {len(episode_rows)} parquet files, found {len(data_files)}")

    total_rows = 0
    for row in tqdm.tqdm(episode_rows, desc="Verifying parquet files"):
        episode_index = row["episode_index"]
        path = staging / _V21_DATA_PATH.format(
            episode_chunk=episode_index // info["chunks_size"],
            episode_index=episode_index,
        )
        table = pq.read_table(path, columns=["episode_index"])
        if table.num_rows != row["length"]:
            raise ValueError(f"{path} has {table.num_rows} rows; expected {row['length']}")
        if set(table.column("episode_index").to_pylist()) != {episode_index}:
            raise ValueError(f"{path} contains an incorrect episode index")
        total_rows += table.num_rows
    if total_rows != info["total_frames"]:
        raise ValueError(f"Converted parquet files contain {total_rows} rows; expected {info['total_frames']}")

    video_jobs = []
    for row in episode_rows:
        episode_index = row["episode_index"]
        for video_key in video_keys:
            path = staging / _V21_VIDEO_PATH.format(
                episode_chunk=episode_index // info["chunks_size"],
                video_key=video_key,
                episode_index=episode_index,
            )
            video_jobs.append((path, row["length"], info["fps"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_verify_video, job) for job in video_jobs]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Verifying videos"):
            future.result()


def convert(source: pathlib.Path, output: pathlib.Path, *, workers: int, verify: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source directory or a child of it")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"Required executable is not installed: {executable}")

    info_path = source / "meta" / "info.json"
    source_info = json.loads(info_path.read_text())
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected a LeRobot v3.0 dataset, got {source_info.get('codebase_version')!r}")

    episode_rows = _load_episode_rows(source)
    tasks = _load_tasks(source)
    if len(episode_rows) != source_info["total_episodes"]:
        raise ValueError(
            f"Metadata contains {len(episode_rows)} episodes; info.json declares {source_info['total_episodes']}"
        )
    if sum(row["length"] for row in episode_rows) != source_info["total_frames"]:
        raise ValueError("Episode lengths do not add up to total_frames")

    video_keys = [key for key, feature in source_info["features"].items() if feature["dtype"] == "video"]
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)

    legacy_info = copy.deepcopy(source_info)
    legacy_info.update(
        {
            "codebase_version": "v2.1",
            "features": _legacy_features(source_info["features"]),
            "total_tasks": len(tasks),
            "total_videos": len(episode_rows) * len(video_keys),
            "total_chunks": math.ceil(len(episode_rows) / source_info["chunks_size"]),
            "splits": {"train": f"0:{len(episode_rows)}"},
            "data_path": _V21_DATA_PATH,
            "video_path": _V21_VIDEO_PATH if video_keys else None,
        }
    )
    legacy_info.pop("data_files_size_in_mb", None)
    legacy_info.pop("video_files_size_in_mb", None)

    feature_names = list(source_info["features"])
    legacy_episodes = [
        {"episode_index": row["episode_index"], "tasks": row["tasks"], "length": row["length"]} for row in episode_rows
    ]
    legacy_episode_stats = [
        {"episode_index": row["episode_index"], "stats": _legacy_episode_stats(row, feature_names)}
        for row in episode_rows
    ]

    _write_json(staging / "meta" / "info.json", legacy_info)
    _write_jsonlines(staging / "meta" / "tasks.jsonl", tasks)
    _write_jsonlines(staging / "meta" / "episodes.jsonl", legacy_episodes)
    _write_jsonlines(staging / "meta" / "episodes_stats.jsonl", legacy_episode_stats)
    _split_parquet_files(source, staging, source_info, episode_rows)
    _split_videos(source, staging, source_info, episode_rows, video_keys, workers)
    if verify:
        _verify_conversion(staging, legacy_info, episode_rows, video_keys, workers)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    print(
        f"Converted {len(episode_rows)} episodes, {legacy_info['total_frames']} frames, "
        f"and {legacy_info['total_videos']} videos to {output}"
    )


def main(
    source: pathlib.Path,
    output: pathlib.Path,
    workers: int = 8,
    *,
    verify: bool = False,
) -> None:
    """Convert a v3 dataset without modifying the source.

    Args:
        source: Local LeRobot v3.0 dataset root.
        output: New LeRobot v2.1 dataset root. It must not already exist.
        workers: Number of concurrent FFmpeg/FFprobe processes.
        verify: Validate every output parquet and video file before publishing the output directory.
    """
    convert(source, output, workers=workers, verify=verify)


if __name__ == "__main__":
    tyro.cli(main)
