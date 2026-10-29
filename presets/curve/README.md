# Default dual-camera curve preset

Approved default: the refined five-minute test `soccer_smooth_curve_5min_v2.mp4` from this conversation.

Use these frozen maps for subsequent renders of the same camera placement and original 3750×1512 stitched panorama geometry. They include the smoother foreground curve and refinements to the left/right junctions and inner white/yellow markings. Apply once to the original panorama, before the curve correction. Do not apply to an already corrected test video.

`render.ps1` applies this default, preserves available audio and checks the encoded output. It defaults to a five-minute test. A full-match render requires an explicit request; pass `-DurationSeconds 0` when authorized.

The maps and settings are frozen here independently of scratch files. `checksums.json` records their hashes. `build_reference.py` records the estimation algorithm; rendering uses the saved maps directly and does not need Python.

Changes to lens, mounting, stabilization, resolution or camera placement require checking alignment and potentially recalibrating. This preset does not make a new camera arrangement automatically match the old field view.
