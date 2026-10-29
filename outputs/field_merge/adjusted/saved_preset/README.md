# Saved settings for this recording

This is the accepted configuration used for field_adjusted.mp4, including the lamp highlight protection and smoother lower-right yellow-line transition.

- DJI: DJI_20261021203051_0002_D.MP4
- GoPro: GOPR0499.MP4
- Timing: DJI seconds = GoPro seconds minus 35.59225.
- Output: 2218x894, 30000/1001 FPS, H.264, GoPro audio.
- Decode without autorotation and resize source frames to 1600x900 before applying the frozen maps.

Files:
- preset.json records source identities, timing, projection, seam and highlight settings.
- frozen_maps.npz preserves the actual fitted camera remaps, local field correction, blend weights, placement and color gains. It avoids refitting or changing the seam on future renders.
- apply_preset.py loads those arrays and composes synchronized BGR frames using NumPy and OpenCV. Its compose method handles source resizing; the caller must decode without autorotation and supply synchronized frames.

Verification: reloading this saved preset and applying it to the same reference source frames reproduced the accepted preview JPEG byte for byte. Source identity checks record byte lengths and SHA-256 of each file's first and last MiB; these are partial fingerprints, not whole-file hashes.

Scope: saved for the listed recording and camera setup. This preserves the approved short-test appearance; it does not establish full-match synchronization drift or geometric stability. Later GoPro chapters require a continuous timeline, and the second DJI recording requires a checked timing relationship. Do not apply this preset to a changed mount, lens, resolution/aspect ratio or unrelated recording without validation.
