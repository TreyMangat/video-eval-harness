# EPIC-KITCHENS subset

This directory contains a deterministic 20-segment EPIC-KITCHENS subset definition for VBench.

## What is already prepared

- `EPIC_100_train.csv` annotations downloaded: yes
- `selected_segments.json` written: yes
- `ground_truth.json` written: yes (20 entries)
- Trimmed segment clips extracted: 0

## Selected source videos

- `P08_05` (P08, 1753.7693329999997s) - missing
- `P22_07` (P22, 2180.6124170000003s) - missing

## Manual completion steps

The official EPIC downloader assets are in `downloader/` and the selected source videos are small in count but still full-length kitchen recordings.

1. From this folder, retry the official downloader with a longer timeout or on a faster/unrestricted network:

```powershell
cd data/epic_kitchens_subset/downloader
py -3.12 epic_downloader.py --videos --specific-videos P08_05,P22_07 --output-path ..\source
```

2. Confirm the source videos exist under `source/EPIC-KITCHENS/<participant>/videos/`.

3. Rerun the subset script without redownloading annotations:

```powershell
cd C:\Users\trey2\Desktop\video_labelling
py -3.12 scripts\download_epic_kitchens_subset.py --skip-video-download
```

## Notes

- The EPIC annotations are public and downloaded successfully from GitHub.
- In this environment, the official Bristol-hosted video download path did not complete the two selected source videos quickly enough for automated extraction.
- The EPIC 2024 challenge site uses registration for challenge participation, but this script uses the official downloader and Bristol dataset endpoints for raw video retrieval.

## Last automated download attempt

- The official EPIC downloader did not finish within 5 seconds for source videos P08_05, P22_07.
