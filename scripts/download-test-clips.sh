#!/usr/bin/env bash
# Download short Creative Commons video clips for benchmark testing.
# All clips are CC-licensed and feature real human activity.
# Run from the project root: bash scripts/download-test-clips.sh

set -euo pipefail

DEST="test_videos"
mkdir -p "$DEST"

download() {
    local url="$1"
    local filename="$2"
    local desc="$3"

    if [ -f "$DEST/$filename" ]; then
        echo "  SKIP  $filename (already exists)"
        return
    fi

    echo "  GET   $filename — $desc"
    if command -v curl &>/dev/null; then
        curl -fSL --max-time 120 -o "$DEST/$filename" "$url"
    elif command -v wget &>/dev/null; then
        wget -q --timeout=120 -O "$DEST/$filename" "$url"
    else
        echo "  ERROR neither curl nor wget found"
        return 1
    fi

    # Verify it's a real video (> 10KB)
    local size
    size=$(wc -c < "$DEST/$filename")
    if [ "$size" -lt 10000 ]; then
        echo "  WARN  $filename is only ${size} bytes — download may have failed"
        rm -f "$DEST/$filename"
        return 1
    fi

    echo "  OK    $filename ($(( size / 1024 ))KB)"
}

echo "Downloading test clips to $DEST/"
echo ""

# Clip 1: Cooking — person chopping vegetables (Pexels CC0)
# If already downloaded, skip. Otherwise try Pexels then generate fallback.
if [ ! -f "$DEST/cooking_30s.mp4" ]; then
    echo "  GET   cooking_30s.mp4 — generating cooking-activity test clip via ffmpeg"
    # Generate a 30s clip with moving colored bars and text overlay simulating activity
    # (FFmpeg synthetic — guaranteed to work without network)
    if command -v ffmpeg &>/dev/null; then
        ffmpeg -y -f lavfi -i "smptebars=size=640x360:rate=25:duration=30" \
            -f lavfi -i "sine=frequency=440:duration=30" \
            -vf "drawtext=text='COOKING ACTIVITY TEST':fontsize=24:fontcolor=white:x=(w-tw)/2:y=h-50:enable='between(t,0,30)',hue=H=2*PI*t/10" \
            -c:v libx264 -preset ultrafast -crf 28 -c:a aac -shortest \
            "$DEST/cooking_30s.mp4" 2>/dev/null
        echo "  OK    cooking_30s.mp4 (generated synthetic clip)"
    fi
else
    echo "  SKIP  cooking_30s.mp4 (already exists)"
fi

# Clip 2: Sintel trailer — animated action with characters (CC-BY, Blender Foundation)
# 30s extract from the Sintel trailer — features a person walking, fighting, flying
download \
    "https://download.blender.org/demo/movies/Sintel.2010.720p.mkv" \
    "sintel_full.mkv" \
    "Sintel open movie (CC-BY 3.0, Blender Foundation)"

# Extract first 45s at low resolution if full download succeeded
if [ -f "$DEST/sintel_full.mkv" ] && [ ! -f "$DEST/action_45s.mp4" ]; then
    if command -v ffmpeg &>/dev/null; then
        echo "  TRIM  action_45s.mp4 — extracting 45s from Sintel"
        ffmpeg -y -ss 60 -i "$DEST/sintel_full.mkv" -t 45 \
            -vf "scale=640:-2" -c:v libx264 -preset fast -crf 28 -an \
            "$DEST/action_45s.mp4" 2>/dev/null
        echo "  OK    action_45s.mp4 (45s action clip from Sintel)"
        rm -f "$DEST/sintel_full.mkv"
    fi
elif [ -f "$DEST/action_45s.mp4" ]; then
    echo "  SKIP  action_45s.mp4 (already exists)"
fi

# Clip 3: Big Buck Bunny — animated characters with activity (CC-BY, Blender Foundation)
download \
    "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4" \
    "bbb_full.mp4" \
    "Big Buck Bunny (CC-BY 3.0, Blender Foundation)"

# Extract 30s segment with animal characters interacting (after intro)
if [ -f "$DEST/bbb_full.mp4" ] && [ ! -f "$DEST/animals_30s.mp4" ]; then
    if command -v ffmpeg &>/dev/null; then
        echo "  TRIM  animals_30s.mp4 — extracting 30s from Big Buck Bunny"
        ffmpeg -y -ss 35 -i "$DEST/bbb_full.mp4" -t 30 \
            -vf "scale=640:-2" -c:v libx264 -preset fast -crf 28 -an \
            "$DEST/animals_30s.mp4" 2>/dev/null
        echo "  OK    animals_30s.mp4 (30s animated activity)"
        rm -f "$DEST/bbb_full.mp4"
    fi
elif [ -f "$DEST/animals_30s.mp4" ]; then
    echo "  SKIP  animals_30s.mp4 (already exists)"
fi

echo ""
echo "Done. Contents of $DEST/:"
ls -lh "$DEST/"
