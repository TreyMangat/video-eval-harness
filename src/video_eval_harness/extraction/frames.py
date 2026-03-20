"""Frame extraction from video segments."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ExtractionConfig
from ..log import get_logger
from ..schemas import ExtractedFrames, Segment
from ..storage import Storage
from ..utils.ffmpeg import extract_frame_at_time

logger = get_logger(__name__)


class FrameExtractor:
    """Extract representative frames from video segments."""

    def __init__(self, config: ExtractionConfig, storage: Storage):
        self.config = config
        self.storage = storage

    def extract(
        self,
        segment: Segment,
        video_path: str | Path,
        num_frames: int | None = None,
        method: str | None = None,
        variant_id: str | None = None,
    ) -> ExtractedFrames:
        """Backward-compatible wrapper for variant-aware extraction."""
        return self.extract_frames(
            segment=segment,
            video_path=video_path,
            num_frames=num_frames,
            method=method,
            variant_id=variant_id,
        )

    def extract_frames(
        self,
        segment: Segment,
        video_path: str | Path,
        num_frames: int | None = None,
        method: str | None = None,
        variant_id: str | None = None,
    ) -> ExtractedFrames:
        """Extract frames for a single segment.

        Frame caches are variant-aware so sweep runs do not overwrite one
        another. Cached frames live at:
        artifacts/frames/<video_id>/<variant_id>/<segment_id>/
        """
        requested_num_frames = num_frames if num_frames is not None else self.config.num_frames
        sampling_method = (method or self.config.method or "uniform").strip().lower()
        effective_variant_id = variant_id or self._default_variant_id(
            requested_num_frames, sampling_method
        )
        frames_dir = self.storage.frames_dir(
            segment.video_id, segment.segment_id, effective_variant_id
        )

        cached = self._load_cached_variant(frames_dir)
        if cached is not None:
            logger.debug(
                "Frames already extracted for %s [%s]",
                segment.segment_id,
                effective_variant_id,
            )
            self.storage.save_extracted_frames(cached)
            return cached

        if sampling_method == "uniform":
            timestamps = self._uniform_timestamps(segment, requested_num_frames)
        elif sampling_method == "keyframe":
            timestamps = self._keyframe_timestamps(
                segment, video_path, requested_num_frames
            )
        else:
            raise ValueError(f"Unknown extraction method '{sampling_method}'")

        frame_paths: list[str] = []
        extracted_timestamps: list[float] = []

        for i, ts in enumerate(timestamps):
            fname = f"frame_{i:03d}.{self.config.image_format}"
            out_path = frames_dir / fname

            try:
                extract_frame_at_time(
                    video_path=video_path,
                    timestamp_s=ts,
                    output_path=out_path,
                    max_dimension=self.config.max_dimension,
                    quality=self.config.image_quality,
                )
                frame_paths.append(str(out_path))
                extracted_timestamps.append(round(ts, 3))
            except Exception as e:
                logger.warning(
                    f"Failed to extract frame at {ts:.2f}s for {segment.segment_id}: {e}"
                )

        result = ExtractedFrames(
            segment_id=segment.segment_id,
            video_id=segment.video_id,
            frame_paths=frame_paths,
            frame_timestamps_s=extracted_timestamps,
            num_frames=len(frame_paths),
        )

        if self.config.generate_contact_sheet and frame_paths:
            try:
                cs_path = self._make_contact_sheet(frame_paths, frames_dir)
                result.contact_sheet_path = str(cs_path)
            except Exception as e:
                logger.warning(f"Contact sheet generation failed: {e}")

        self._write_variant_manifest(frames_dir, result)
        self.storage.save_extracted_frames(result)
        return result

    def _uniform_timestamps(self, segment: Segment, num_frames: int) -> list[float]:
        duration = segment.duration_s

        if num_frames <= 0:
            return []

        if num_frames == 1:
            return [segment.start_time_s + duration / 2]

        margin = duration * 0.05
        effective_start = segment.start_time_s + margin
        effective_end = segment.end_time_s - margin
        if effective_end <= effective_start:
            effective_start = segment.start_time_s
            effective_end = segment.end_time_s

        step = (effective_end - effective_start) / (num_frames - 1)
        return [effective_start + i * step for i in range(num_frames)]

    def _keyframe_timestamps(
        self, segment: Segment, video_path: str | Path, num_frames: int
    ) -> list[float]:
        if num_frames <= 0:
            return []

        if num_frames == 1:
            return self._uniform_timestamps(segment, 1)

        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV unavailable; falling back to uniform extraction.")
            return self._uniform_timestamps(segment, num_frames)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Could not open %s for keyframe extraction; using uniform.", video_path)
            return self._uniform_timestamps(segment, num_frames)

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 0:
                return self._uniform_timestamps(segment, num_frames)

            start_frame = max(0, int(segment.start_time_s * fps))
            end_frame = max(start_frame + 1, int(segment.end_time_s * fps))
            sample_interval = max(1, int(round(fps / 4.0)))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            prev_hist = None
            candidates: list[tuple[float, float]] = []

            while frame_idx < end_frame:
                ok, frame = cap.read()
                if not ok:
                    break

                if (frame_idx - start_frame) % sample_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                    cv2.normalize(hist, hist)
                    if prev_hist is None:
                        diff = 1.0
                    else:
                        similarity = cv2.compareHist(
                            prev_hist, hist, cv2.HISTCMP_CORREL
                        )
                        diff = 1.0 - similarity
                    candidates.append((diff, round(frame_idx / fps, 3)))
                    prev_hist = hist

                frame_idx += 1

            if not candidates:
                return self._uniform_timestamps(segment, num_frames)

            ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
            selected: list[float] = []
            for _, ts in ranked:
                if ts not in selected:
                    selected.append(ts)
                if len(selected) == min(num_frames, len(candidates)):
                    break

            if len(selected) < num_frames:
                for ts in self._uniform_timestamps(segment, num_frames):
                    rounded = round(ts, 3)
                    if rounded not in selected:
                        selected.append(rounded)
                    if len(selected) == num_frames:
                        break

            return sorted(selected[:num_frames])
        finally:
            cap.release()

    def _load_cached_variant(self, frames_dir: Path) -> ExtractedFrames | None:
        manifest_path = frames_dir / "metadata.json"
        if not manifest_path.exists():
            return None

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        frame_paths = payload.get("frame_paths") or []
        if not frame_paths:
            return None
        if not all(Path(path_str).exists() for path_str in frame_paths):
            return None

        contact_sheet_path = payload.get("contact_sheet_path")
        if contact_sheet_path and not Path(contact_sheet_path).exists():
            contact_sheet_path = None

        return ExtractedFrames(
            segment_id=payload["segment_id"],
            video_id=payload["video_id"],
            frame_paths=frame_paths,
            frame_timestamps_s=payload.get("frame_timestamps_s") or [],
            contact_sheet_path=contact_sheet_path,
            num_frames=int(payload.get("num_frames") or len(frame_paths)),
        )

    def _write_variant_manifest(self, frames_dir: Path, frames: ExtractedFrames) -> None:
        manifest_path = frames_dir / "metadata.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "segment_id": frames.segment_id,
                    "video_id": frames.video_id,
                    "frame_paths": frames.frame_paths,
                    "frame_timestamps_s": frames.frame_timestamps_s,
                    "contact_sheet_path": frames.contact_sheet_path,
                    "num_frames": frames.num_frames,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _default_variant_id(self, num_frames: int, method: str) -> str:
        return f"{method}_{num_frames}f"

    def _make_contact_sheet(self, frame_paths: list[str], output_dir: Path) -> Path:
        """Create a tiled contact sheet image from extracted frames."""
        from PIL import Image

        images = [Image.open(p) for p in frame_paths]
        if not images:
            raise ValueError("No images for contact sheet")

        cols = self.config.contact_sheet_cols
        rows = (len(images) + cols - 1) // cols

        w, h = images[0].size
        tile_w = min(w, 320)
        tile_h = int(tile_w * h / w)

        sheet = Image.new("RGB", (tile_w * cols, tile_h * rows), (0, 0, 0))
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            resized = img.resize((tile_w, tile_h))
            sheet.paste(resized, (c * tile_w, r * tile_h))

        out_path = output_dir / "contact_sheet.jpg"
        sheet.save(out_path, quality=85)
        return out_path
