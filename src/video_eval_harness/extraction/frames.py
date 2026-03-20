"""Frame extraction from video segments."""

from __future__ import annotations

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

    def extract(self, segment: Segment, video_path: str | Path) -> ExtractedFrames:
        """Extract frames for a single segment.

        Uses uniform sampling across the segment duration.
        """
        # Check if already extracted
        existing = self.storage.get_extracted_frames(segment.segment_id)
        if existing and existing.num_frames > 0:
            # Verify files still exist
            if all(Path(p).exists() for p in existing.frame_paths):
                logger.debug(f"Frames already extracted for {segment.segment_id}")
                return existing

        frames_dir = self.storage.frames_dir(segment.video_id, segment.segment_id)
        n = self.config.num_frames
        duration = segment.duration_s

        # Calculate timestamps for uniform sampling
        if n == 1:
            timestamps = [segment.start_time_s + duration / 2]
        else:
            # Spread frames evenly, avoiding exact start/end boundaries
            margin = duration * 0.05  # 5% margin
            effective_start = segment.start_time_s + margin
            effective_end = segment.end_time_s - margin
            if effective_end <= effective_start:
                effective_start = segment.start_time_s
                effective_end = segment.end_time_s
            step = (effective_end - effective_start) / (n - 1) if n > 1 else 0
            timestamps = [effective_start + i * step for i in range(n)]

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
                logger.warning(f"Failed to extract frame at {ts:.2f}s for {segment.segment_id}: {e}")

        result = ExtractedFrames(
            segment_id=segment.segment_id,
            video_id=segment.video_id,
            frame_paths=frame_paths,
            frame_timestamps_s=extracted_timestamps,
            num_frames=len(frame_paths),
        )

        # Generate contact sheet if enabled
        if self.config.generate_contact_sheet and frame_paths:
            try:
                cs_path = self._make_contact_sheet(frame_paths, frames_dir)
                result.contact_sheet_path = str(cs_path)
            except Exception as e:
                logger.warning(f"Contact sheet generation failed: {e}")

        self.storage.save_extracted_frames(result)
        return result

    def _make_contact_sheet(self, frame_paths: list[str], output_dir: Path) -> Path:
        """Create a tiled contact sheet image from extracted frames."""
        from PIL import Image

        images = [Image.open(p) for p in frame_paths]
        if not images:
            raise ValueError("No images for contact sheet")

        cols = self.config.contact_sheet_cols
        rows = (len(images) + cols - 1) // cols

        # Determine tile size from first image
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
