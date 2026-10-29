# Full higher-quality panorama

## Scope
The soccer recordings are stitched into one panorama using the accepted higher-resolution settings. The unrelated indoor test files are excluded. Source GoPro time 50.000 through 3450.000 seconds is used, yielding approximately 56 minutes 40 seconds. Earlier material lacks stable paired coverage; late material shows the rig being picked up. Original recordings remain unchanged.

## Sources and timing
Five GoPro chapters (GOPR0499, GP010499, GP020499, GP030499, GP040499) and two DJI files (DJI_20261021203051_0002_D and DJI_20261021211107_0003_D) are used. GoPro chapter positions are based on video duration, not AAC packet/container padding.

Eighteen independently located audio correspondences across the recording have normalized correlation coefficients approximately 0.41–0.70. A separate linear clock relationship is fitted for each camera-file/chapter intersection. This compensates gradual clock drift and small chapter-boundary timing changes rather than extrapolating the original fixed offset across the entire recording. Exact anchors and models are saved in timeline_probe.json and render_plan.json.

The inferred DJI file change leaves approximately 0.047 seconds without a new DJI video frame. The final preceding DJI frame is held briefly while GoPro video/audio continue. No match time is removed at this join. Output frames lie on a continuous 30000/1001-FPS grid; small boundary rounding is handled by source-frame duplication where needed.

## Image processing
Approved frozen spherical projection, empirical local field correction, fixed seam and color settings are retained. The lamp highlight protection and smoother yellow-line transition remain enabled. Source frames are decoded at 2704x1520; output is 3750x1512. GPU bicubic inverse mapping samples directly from these larger source frames. There is no AI upscaling, tracking or automatic camera movement.

NVIDIA H.264 p6/CQ17 is used, matching the approved higher-quality test. Recoverable sections are encoded separately and joined by stream copy, avoiding another video encode. GoPro AAC audio is made continuous across chapter packet boundaries. Black borders preserve warped coverage. Sources are not overwritten.

## Verification and limits
Preflight panoramas were inspected throughout the stable recording and around the camera-file transition. Setup and pack-up frames were separately inspected to choose the usable interval. Each encoded section is checked for its expected packet count and dimensions. The completed MP4 is fully decoded with error-stop checking and checked against the planned total frame count before delivery.

This preserves the accepted empirical alignment; it is not a physical lens calibration. Curvature, occasional player parallax/ghosting, some color variation and source night noise remain possible. Timing anchors establish local audio correspondence, not exact simultaneous sensor exposures. Completion status and final file measurements will be recorded in completion.json and final_probe.json.

Final metadata check: 101,898 frames; 3750x1512; approximately 29.97 FPS; 3399.997 seconds video and 3400.006 seconds container (56:40); 21,353,460,245 bytes. Final audio comparisons at output seconds 30, 1400, 2800 and 3300 matched the corresponding GoPro source audio with correlation 0.979–1.000 and residual offsets 0, 16.625, 12 and 12 milliseconds. These include chapter assembly and final mux timing. The frames immediately before/after the DJI file change were also visually inspected and retain consistent framing.

Completion: full-file FFmpeg decoding with error-stop checking passed. Every encoded section and the final MP4 have the expected frame count. Rendering, audio assembly, joining and full decoding took approximately 73 minutes, excluding preflight and the final copy. Final validation details are saved in completion.json, final_probe.json and final_audio_checks.json.
