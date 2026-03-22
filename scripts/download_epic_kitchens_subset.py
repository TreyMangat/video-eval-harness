from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ANNOTATIONS_URL: Final[str] = (
    "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/"
    "master/EPIC_100_train.csv"
)
VIDEO_INFO_URL: Final[str] = (
    "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/"
    "master/EPIC_100_video_info.csv"
)
DOWNLOADER_BASE_URL: Final[str] = (
    "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-download-scripts/master"
)
DOWNLOADER_FILES: Final[dict[str, str]] = {
    "epic_downloader.py": f"{DOWNLOADER_BASE_URL}/epic_downloader.py",
    "data/epic_55_splits.csv": f"{DOWNLOADER_BASE_URL}/data/epic_55_splits.csv",
    "data/epic_100_splits.csv": f"{DOWNLOADER_BASE_URL}/data/epic_100_splits.csv",
    "data/md5.csv": f"{DOWNLOADER_BASE_URL}/data/md5.csv",
    "data/errata.csv": f"{DOWNLOADER_BASE_URL}/data/errata.csv",
}
TARGET_VERBS: Final[list[str]] = [
    "take",
    "put",
    "open",
    "close",
    "wash",
    "cut",
    "mix",
    "pour",
    "turn-on",
    "peel",
]
GERUND_BY_VERB: Final[dict[str, str]] = {
    "take": "taking",
    "put": "putting",
    "open": "opening",
    "close": "closing",
    "wash": "washing",
    "cut": "cutting",
    "mix": "mixing",
    "pour": "pouring",
    "turn-on": "turning on",
    "peel": "peeling",
}
PREFERRED_VIDEOS: Final[list[str]] = ["P08_05", "P22_07", "P22_16", "P01_05", "P01_09"]


@dataclass(frozen=True)
class SelectedSegment:
    clip_id: str
    clip_filename: str
    narration_id: str
    participant_id: str
    video_id: str
    narration: str
    primary_action: str
    verb: str
    noun: str
    start_timestamp: str
    stop_timestamp: str
    start_seconds: float
    stop_seconds: float

    @property
    def segment_id(self) -> str:
        return f"{self.clip_id}_seg000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a small EPIC-KITCHENS subset for VBench: download annotations, "
            "select 20 target segments, optionally fetch source videos, trim clips, "
            "and generate ground_truth.json."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/epic_kitchens_subset"),
        help="Output directory for the subset artifacts.",
    )
    parser.add_argument(
        "--download-timeout-sec",
        type=int,
        default=300,
        help="Timeout for the official EPIC downloader subprocess.",
    )
    parser.add_argument(
        "--skip-video-download",
        action="store_true",
        help="Only prepare annotations, selection metadata, ground truth, and README.",
    )
    return parser.parse_args()


def ensure_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary not found on PATH: {name}")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def maybe_download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    download_file(url, destination)


def parse_timestamp_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def is_valid_video(video_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode == 0 and bool(result.stdout.strip())


def normalize_narration(value: str) -> str:
    return " ".join(value.strip().lower().split())


def narration_quality(row: dict[str, str], verb: str) -> tuple[int, int]:
    narration = normalize_narration(row["narration"])
    token_count = len(narration.split())
    generic_penalty = 1 if token_count <= 1 else 0

    if verb == "turn-on":
        if narration.startswith("turn on"):
            return (0, generic_penalty)
        if narration.startswith("turn "):
            return (1, generic_penalty)
        return (2, generic_penalty)

    if narration.startswith(verb) or narration.startswith(GERUND_BY_VERB[verb]):
        return (0, generic_penalty)
    return (1, generic_penalty)


def candidate_sort_key(row: dict[str, str], verb: str) -> tuple[int, int, float, str, str]:
    video_rank = PREFERRED_VIDEOS.index(row["video_id"]) if row["video_id"] in PREFERRED_VIDEOS else 999
    quality_rank, generic_penalty = narration_quality(row, verb)
    duration_s = parse_timestamp_to_seconds(row["stop_timestamp"]) - parse_timestamp_to_seconds(
        row["start_timestamp"]
    )
    return (
        video_rank,
        quality_rank + generic_penalty,
        duration_s,
        row["start_timestamp"],
        row["narration_id"],
    )


def narration_to_gerund(narration: str, verb: str) -> str:
    normalized = normalize_narration(narration)
    gerund = GERUND_BY_VERB[verb]

    if normalized.startswith(gerund):
        return normalized

    if verb == "turn-on":
        if normalized.startswith("turn on "):
            return f"turning on {normalized[len('turn on '):]}".strip()
        if normalized == "turn on":
            return "turning on"
        if normalized.startswith("turn "):
            return f"turning {normalized[len('turn '):]}".strip()
        if normalized.startswith("open "):
            return f"turning on {normalized[len('open '):]}".strip()
        return "turning on"

    tokens = normalized.split()
    if not tokens:
        return gerund
    if tokens[0] == verb:
        return " ".join([gerund, *tokens[1:]]).strip()
    return " ".join([gerund, *tokens[1:]]).strip()


def select_segments(annotation_rows: list[dict[str, str]]) -> list[SelectedSegment]:
    selected: list[SelectedSegment] = []
    used_narration_ids: set[str] = set()

    for verb in TARGET_VERBS:
        candidates = [
            row
            for row in annotation_rows
            if row["verb"] == verb
            and (parse_timestamp_to_seconds(row["stop_timestamp"]) - parse_timestamp_to_seconds(row["start_timestamp"])) > 0.2
            and (parse_timestamp_to_seconds(row["stop_timestamp"]) - parse_timestamp_to_seconds(row["start_timestamp"])) <= 12
        ]
        candidates.sort(key=lambda row: candidate_sort_key(row, verb))

        picks: list[dict[str, str]] = []
        used_videos: set[str] = set()
        for row in candidates:
            if row["narration_id"] in used_narration_ids or row["video_id"] in used_videos:
                continue
            picks.append(row)
            used_narration_ids.add(row["narration_id"])
            used_videos.add(row["video_id"])
            if len(picks) == 2:
                break

        if len(picks) < 2:
            for row in candidates:
                if row["narration_id"] in used_narration_ids:
                    continue
                picks.append(row)
                used_narration_ids.add(row["narration_id"])
                if len(picks) == 2:
                    break

        if len(picks) != 2:
            raise RuntimeError(f"Could not find two EPIC-KITCHENS segments for verb '{verb}'.")

        for index, row in enumerate(picks, start=1):
            clip_id = f"{row['video_id']}_{verb}_{index:03d}"
            selected.append(
                SelectedSegment(
                    clip_id=clip_id,
                    clip_filename=f"{clip_id}.mp4",
                    narration_id=row["narration_id"],
                    participant_id=row["participant_id"],
                    video_id=row["video_id"],
                    narration=row["narration"],
                    primary_action=narration_to_gerund(row["narration"], verb),
                    verb=verb,
                    noun=row["noun"],
                    start_timestamp=row["start_timestamp"],
                    stop_timestamp=row["stop_timestamp"],
                    start_seconds=parse_timestamp_to_seconds(row["start_timestamp"]),
                    stop_seconds=parse_timestamp_to_seconds(row["stop_timestamp"]),
                )
            )

    return selected


def write_selection_manifest(output_dir: Path, selected_segments: list[SelectedSegment]) -> Path:
    selection_path = output_dir / "selected_segments.json"
    payload = [
        {
            "clip_id": segment.clip_id,
            "clip_filename": segment.clip_filename,
            "narration_id": segment.narration_id,
            "participant_id": segment.participant_id,
            "video_id": segment.video_id,
            "verb": segment.verb,
            "noun": segment.noun,
            "narration": segment.narration,
            "primary_action": segment.primary_action,
            "start_timestamp": segment.start_timestamp,
            "stop_timestamp": segment.stop_timestamp,
            "start_seconds": round(segment.start_seconds, 3),
            "stop_seconds": round(segment.stop_seconds, 3),
        }
        for segment in selected_segments
    ]
    selection_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return selection_path


def write_ground_truth(output_dir: Path, selected_segments: list[SelectedSegment]) -> Path:
    ground_truth_path = output_dir / "ground_truth.json"
    payload = [
        {
            "video_id": segment.clip_id,
            "segment_id": segment.segment_id,
            "primary_action": segment.primary_action,
        }
        for segment in selected_segments
    ]
    ground_truth_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return ground_truth_path


def ensure_downloader_assets(downloader_dir: Path) -> Path:
    for relative_path, url in DOWNLOADER_FILES.items():
        maybe_download_file(url, downloader_dir / relative_path)
    return downloader_dir / "epic_downloader.py"


def download_source_videos(
    output_dir: Path,
    video_ids: list[str],
    timeout_sec: int,
) -> tuple[dict[str, Path], str | None]:
    downloader_dir = output_dir / "downloader"
    downloader_path = ensure_downloader_assets(downloader_dir)
    source_root = output_dir / "source"
    command = [
        sys.executable,
        str(downloader_path.name),
        "--videos",
        "--specific-videos",
        ",".join(video_ids),
        "--output-path",
        "source",
    ]

    error_note: str | None = None
    try:
        subprocess.run(
            command,
            cwd=downloader_dir,
            check=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        error_note = (
            f"The official EPIC downloader did not finish within {timeout_sec} seconds "
            f"for source videos {', '.join(video_ids)}."
        )
    except subprocess.CalledProcessError as exc:
        error_note = f"The official EPIC downloader exited with code {exc.returncode}."

    video_lookup: dict[str, Path] = {}
    for video_id in video_ids:
        participant_id = video_id.split("_")[0]
        candidate_paths = [
            source_root / "EPIC-KITCHENS" / participant_id / "videos" / f"{video_id}.MP4",
            source_root / "EPIC-KITCHENS" / participant_id / "videos" / f"{video_id}.mp4",
        ]
        for path in candidate_paths:
            if path.exists() and path.stat().st_size > 0 and is_valid_video(path):
                video_lookup[video_id] = path
                break
            if path.exists():
                path.unlink(missing_ok=True)

    return video_lookup, error_note


def extract_segment(full_video: Path, segment: SelectedSegment, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)

    copy_command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        segment.start_timestamp,
        "-to",
        segment.stop_timestamp,
        "-i",
        str(full_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        str(destination),
    ]
    result = subprocess.run(copy_command, check=False)
    if result.returncode == 0 and destination.exists() and destination.stat().st_size > 0:
        return True

    if destination.exists():
        destination.unlink(missing_ok=True)

    reencode_command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        segment.start_timestamp,
        "-to",
        segment.stop_timestamp,
        "-i",
        str(full_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = subprocess.run(reencode_command, check=False)
    return result.returncode == 0 and destination.exists() and destination.stat().st_size > 0


def write_readme(
    output_dir: Path,
    selected_segments: list[SelectedSegment],
    video_lookup: dict[str, Path],
    download_note: str | None,
    video_info_lookup: dict[str, dict[str, str]],
) -> Path:
    readme_path = output_dir / "README.md"
    source_videos = sorted({segment.video_id for segment in selected_segments})
    extracted_count = len(list((output_dir / "segments").glob("*.mp4")))

    lines = [
        "# EPIC-KITCHENS subset",
        "",
        "This directory contains a deterministic 20-segment EPIC-KITCHENS subset definition for VBench.",
        "",
        "## What is already prepared",
        "",
        f"- `EPIC_100_train.csv` annotations downloaded: yes",
        f"- `selected_segments.json` written: yes",
        f"- `ground_truth.json` written: yes ({len(selected_segments)} entries)",
        f"- Trimmed segment clips extracted: {extracted_count}",
        "",
        "## Selected source videos",
        "",
    ]

    for video_id in source_videos:
        info = video_info_lookup.get(video_id, {})
        duration = info.get("duration", "unknown")
        participant = info.get("participant_id", video_id.split("_")[0])
        availability = "downloaded" if video_id in video_lookup else "missing"
        lines.append(f"- `{video_id}` ({participant}, {duration}s) - {availability}")

    lines.extend(
        [
            "",
            "## Manual completion steps",
            "",
            "The official EPIC downloader assets are in `downloader/` and the selected source videos are small in count but still full-length kitchen recordings.",
            "",
            "1. From this folder, retry the official downloader with a longer timeout or on a faster/unrestricted network:",
            "",
            "```powershell",
            "cd data/epic_kitchens_subset/downloader",
            f'py -3.12 epic_downloader.py --videos --specific-videos {",".join(source_videos)} --output-path ..\\source',
            "```",
            "",
            "2. Confirm the source videos exist under `source/EPIC-KITCHENS/<participant>/videos/`.",
            "",
            "3. Rerun the subset script without redownloading annotations:",
            "",
            "```powershell",
            "cd C:\\Users\\trey2\\Desktop\\video_labelling",
            "py -3.12 scripts\\download_epic_kitchens_subset.py --skip-video-download",
            "```",
            "",
            "## Notes",
            "",
            "- The EPIC annotations are public and downloaded successfully from GitHub.",
            "- In this environment, the official Bristol-hosted video download path did not complete the two selected source videos quickly enough for automated extraction.",
            "- The EPIC 2024 challenge site uses registration for challenge participation, but this script uses the official downloader and Bristol dataset endpoints for raw video retrieval.",
        ]
    )

    if download_note:
        lines.extend(["", "## Last automated download attempt", "", f"- {download_note}"])

    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    ensure_binary("ffmpeg")
    ensure_binary("ffprobe")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = output_dir / "EPIC_100_train.csv"
    video_info_path = output_dir / "EPIC_100_video_info.csv"
    maybe_download_file(ANNOTATIONS_URL, annotations_path)
    maybe_download_file(VIDEO_INFO_URL, video_info_path)

    annotation_rows = load_csv_rows(annotations_path)
    video_info_rows = load_csv_rows(video_info_path)
    video_info_lookup = {row["video_id"]: row for row in video_info_rows}

    selected_segments = select_segments(annotation_rows)
    selection_path = write_selection_manifest(output_dir, selected_segments)
    ground_truth_path = write_ground_truth(output_dir, selected_segments)

    source_videos = sorted({segment.video_id for segment in selected_segments})
    video_lookup: dict[str, Path] = {}
    download_note: str | None = None

    if not args.skip_video_download:
        video_lookup, download_note = download_source_videos(
            output_dir,
            source_videos,
            args.download_timeout_sec,
        )

    extracted = 0
    for segment in selected_segments:
        source_video = video_lookup.get(segment.video_id)
        if source_video is None:
            continue
        destination = segments_dir / segment.clip_filename
        if extract_segment(source_video, segment, destination):
            extracted += 1

    readme_path = write_readme(
        output_dir,
        selected_segments,
        video_lookup,
        download_note,
        video_info_lookup,
    )

    print(f"Annotations: {annotations_path}")
    print(f"Selection manifest: {selection_path}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Trimmed segments extracted: {extracted}/{len(selected_segments)}")
    print(f"README: {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
