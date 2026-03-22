# EPIC-KITCHENS subset

This directory contains a deterministic 20-segment EPIC-KITCHENS subset for VBench.

## Status

- `EPIC_100_train.csv` annotations downloaded: yes
- `selected_segments.json` written: yes
- `ground_truth.json` written: yes (20 entries)
- Real segment clips extracted: yes (20/20)
- Synthetic fallback needed: no

## What we tried

### Bristol direct / official downloader

- The official downloader assets were fetched successfully.
- The prior downloader attempt left an incomplete `P08_05.MP4` under `downloader/source/`.
- `ffprobe` reported `moov atom not found`, so that file is not usable.
- Direct Bristol URL checks for `P01_05`, `P08_05`, and `P22_07` returned `404` from this environment.

### Academic Torrents

- The Academic Torrents browse endpoint redirected to a browser-check page.
- Their RSS feed/database hint did not surface an obvious EPIC-KITCHENS match during this attempt.
- No usable torrent or mirror link was recovered here.

### Hugging Face mirrors

- Hugging Face dataset search returned multiple EPIC-KITCHENS mirrors.
- `lightly-ai/epic-kitchens-100-clips` did not contain this subset's 20 selected `narration_id` clips.
- `a1raman/epic_kitchens_100` exposed the exact source videos we needed:
  - `P08/videos/P08_05.MP4`
  - `P22/videos/P22_07.MP4`
- Those full videos are large, but `ffmpeg` was able to seek against the remote Hugging Face URLs and extract only the required short segments.

## What worked

- Public EPIC annotations downloaded successfully.
- Ground-truth generation completed successfully.
- Remote segment extraction from the Hugging Face mirror completed successfully.
- The final subset now contains 20 real EPIC-KITCHENS `.mp4` clips in `segments/`.

## Segment source videos

- `P08_05`: 10 extracted clips
- `P22_07`: 10 extracted clips

## Manual completion options

If you want to rebuild the subset on another machine with better network access, you have two practical paths:

1. Use the Hugging Face mirror and remote extraction:

```powershell
ffmpeg -y -loglevel error -ss 00:16:12.79 -to 00:16:13.74 `
  -i "https://huggingface.co/datasets/a1raman/epic_kitchens_100/resolve/main/P08/videos/P08_05.MP4?download=true" `
  -map 0:v:0 -map 0:a? -c:v libx264 -preset veryfast -crf 24 -pix_fmt yuv420p -c:a aac `
  -movflags +faststart data/epic_kitchens_subset/segments/P08_05_take_001.mp4
```

2. Or retry the official downloader for full source videos:

```powershell
cd data/epic_kitchens_subset/downloader
py -3.12 epic_downloader.py --videos --specific-videos P08_05,P22_07 --output-path ..\source
```

If the full source videos finish downloading successfully, you can trim the selected clips with the timestamps in `selected_segments.json`.
