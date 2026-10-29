param(
    [Parameter(Mandatory=$true)][string]$InputVideo,
    [Parameter(Mandatory=$true)][string]$OutputVideo,
    [double]$DurationSeconds = 300
)
$ErrorActionPreference = 'Stop'
$probe = & ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json $InputVideo | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect input video' }
if ($probe.streams[0].width -ne 3750 -or $probe.streams[0].height -ne 1512) {
    throw 'This saved preset requires the original 3750x1512 stitched panorama geometry.'
}
$renderArgs = @('-v','error','-n','-i',$InputVideo,'-i',(Join-Path $PSScriptRoot 'map_x.pgm'),'-i',(Join-Path $PSScriptRoot 'map_y.pgm'),'-filter_complex','[0:v][1:v][2:v]remap[v]','-map','[v]','-map','0:a?')
if ($DurationSeconds -gt 0) { $renderArgs += @('-t',"$DurationSeconds") }
$renderArgs += @('-c:v','h264_nvenc','-preset','p6','-cq','17','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',$OutputVideo)
& ffmpeg @renderArgs
if ($LASTEXITCODE -ne 0) { throw 'Render failed' }
& ffmpeg -v error -xerror -i $OutputVideo -f null -
if ($LASTEXITCODE -ne 0) { throw 'Output decode check failed' }
