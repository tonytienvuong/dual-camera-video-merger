# Dual-camera soccer video merger

Portable copy of the DJI Osmo Action 4 / GoPro HERO4 merger and approved calibration from October 21–22, 2026. Includes the resumable full-match renderer, measured chapter timing anchors, frozen stitch maps, and the later curve correction preset.

## Windows setup

Install Python 3.12, Git, FFmpeg (including ffprobe) on PATH, and the current NVIDIA graphics driver. The full-match renderer requires NVIDIA H.264 NVENC and an OpenCL GPU; there is no automatic CPU fallback.

```powershell
git clone https://github.com/tonytienvuong/dual-camera-video-merger.git
cd dual-camera-video-merger
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:SOCCER_SOURCE_DIR = 'D:\Soccer\Originals'
ffmpeg -hide_banner -encoders | Select-String h264_nvenc
python -c "import pyopencl as cl; print([(p.name, [d.name for d in p.get_devices()]) for p in cl.get_platforms()])"
```

Copy the original camera recordings separately to that source folder. They are too large for this code repository. Required filenames:

- DJI_20261021203051_0002_D.MP4
- DJI_20261021211107_0003_D.MP4
- GOPR0499.MP4
- GP010499.MP4
- GP020499.MP4
- GP030499.MP4
- GP040499.MP4

## Render the recorded match

Run from the repository root. Check the plan first, then render:

```powershell
python work/render_full_match.py --plan-only
python work/render_full_match.py
```

Output: `outputs/field_merge/full_match/final_panorama.mp4` (3750x1512, about 56:40). Allow substantial free disk space for intermediate sections, the panorama, and corrected output. The original completed panorama was about 21 GB. Timing anchors and the 50–3450 second interval are specific to these recordings, not an automatic solution for a new match.

Rendering reuses completed sections in `work/full_match_parts`. To request a stop, create `work/stop_full_match.flag`; remove it before resuming. A final file that already exists is never overwritten. Check `outputs/field_merge/full_match/progress.json` for progress. FFmpeg performs a complete decode check after rendering.

## Apply the approved curve correction

```powershell
powershell -ExecutionPolicy Bypass -File presets/curve/render.ps1 -InputVideo "$PWD\outputs\field_merge\full_match\final_panorama.mp4" -OutputVideo "$PWD\corrected-test.mp4"
```

This defaults to a five-minute test. After reviewing it, use a different output filename and `-DurationSeconds 0` to correct the entire match. Apply the maps once to the original stitched panorama. This preset was accepted for this camera placement; a new placement needs alignment and calibration checks.

## Included history and limitations

See `outputs/field_merge/full_match/report.md` for the original successful full-match validation and remaining visual limitations. That report describes the original computer's run, not a new render here. The transfer preserves the existing prototype architecture, including dynamic loading of shared rendering functions. This is a fixed-camera panorama merger, with no automatic player tracking or highlight detection.
