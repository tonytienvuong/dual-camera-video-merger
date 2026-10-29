"""Render native-GoPro-resolution sources through the approved frozen warp."""
import sys,json,time,subprocess,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np
from numba import njit,prange,set_num_threads
base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base/'outputs/field_merge/adjusted/saved_preset'))
from apply_preset import FrozenStitchPreset
cv2.setNumThreads(4);set_num_threads(4)
preset=FrozenStitchPreset();d=preset.data
out=base/'outputs/field_merge/high_quality';out.mkdir(exist_ok=True)
source_size=(2704,1520);scale=2704/1600
w=int(math.ceil(preset.width*scale/2)*2);h=int(math.ceil(preset.height*scale/2)*2)
gx,gy=np.meshgrid((np.arange(w,dtype=np.float32)+.5)/scale-.5,(np.arange(h,dtype=np.float32)+.5)/scale-.5)
maps=[]
for i in range(2):
    fullx=np.full((preset.height,preset.width),-1,np.float32);fully=fullx.copy()
    y0,y1,x0,x1=map(int,d[f'placement_{i}'])
    fullx[y0:y1,x0:x1]=d[f'map_x_{i}'];fully[y0:y1,x0:x1]=d[f'map_y_{i}']
    if i==1:
        fullx=cv2.remap(fullx,d['rx'],d['ry'],cv2.INTER_LINEAR,borderValue=-1)
        fully=cv2.remap(fully,d['rx'],d['ry'],cv2.INTER_LINEAR,borderValue=-1)
    mx=cv2.remap(fullx,gx,gy,cv2.INTER_LINEAR,borderValue=-1)
    my=cv2.remap(fully,gx,gy,cv2.INTER_LINEAR,borderValue=-1)
    mx=(mx+.5)*(2704/1600)-.5;my=(my+.5)*(1520/900)-.5
    maps.append((mx,my))
weights=[cv2.remap(d[f'weight_{i}'][:,:,0],gx,gy,cv2.INTER_LINEAR) for i in range(2)]
total=np.maximum(weights[0]+weights[1],1e-6)
weights=[(wt/total)[:,:,None] for wt in weights]
del gx,gy,fullx,fully,total
# Reuse the verified fused color/highlight/blending implementation.
source=(Path(__file__).parent/'render_five_minutes.py').read_text()
exec(source[source.index('@njit(parallel=True)\ndef fused'):source.index('def fast(frames):')])
def compose(frames):
    ims=[cv2.remap(im,mx,my,cv2.INTER_LANCZOS4).astype(np.float32) for im,(mx,my) in zip(frames,maps)]
    return fused(*ims,d['gains'],d['half_gain'],*weights)
if '--gpu' in sys.argv:
    from gpu_composer import GpuComposer
    cpu_compose=compose
    gpu=GpuComposer(maps,weights,d['gains'],d['half_gain'],(h,w,3))
    print('GPU',gpu.name,flush=True)
    compose=gpu.compose
paths=[Path(__import__('os').environ.get('SOCCER_SOURCE_DIR', r'F:\Video\test')) / Path(s['path'].replace(chr(92), '/')).name for s in preset.config['sources']]
start=85.;duration=3. if '--sample' in sys.argv else 300.
target=out/(('hq_gpu_cubic_sample.mp4' if '--gpu' in sys.argv else 'hq_sample.mp4') if '--sample' in sys.argv else 'soccer_panorama_5min_hq.mp4')
fps=30000/1001;frames_target=round(duration*fps)
readers=[]
for p,t in zip(paths,[start+preset.config['offset_dji_minus_gopro'],start]):
    readers.append(subprocess.Popen(list(map(str,['ffmpeg','-v','error','-noautorotate','-ss',t,'-i',p,'-frames:v',frames_target,'-map','0:v:0','-vf','scale=2704:1520:flags=lanczos','-pix_fmt','bgr24','-f','rawvideo','-'])),stdout=subprocess.PIPE))
enc=subprocess.Popen(list(map(str,['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','30000/1001','-i','-','-ss',start,'-i',paths[1],'-map','0:v:0','-map','1:a:0','-t',duration,'-c:v','h264_nvenc','-preset','p6','-cq','17','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',target])),stdin=subprocess.PIPE)
count=0;tick=time.perf_counter()
print('START',w,h,frames_target,flush=True)
try:
    while count<frames_target:
        chunks=[r.stdout.read(2704*1520*3) for r in readers]
        if any(len(c)!=2704*1520*3 for c in chunks):raise RuntimeError('Source ended early')
        ims=[np.frombuffer(c,np.uint8).reshape(1520,2704,3) for c in chunks]
        result=compose(ims)
        if count==0 and '--gpu' in sys.argv and '--sample' in sys.argv:
            expected=cpu_compose(ims);delta=np.abs(result.astype(np.int16)-expected.astype(np.int16))
            print('GPU_COMPARISON mean/p99',float(delta.mean()),float(np.percentile(delta,99)),flush=True)
            assert delta.mean()<1.2 and np.percentile(delta,99)<5,'GPU sampling differs excessively'
        if count==0:cv2.imwrite(str(out/('hq_sample_preview.jpg' if '--sample' in sys.argv else 'hq_first_frame.jpg')),result,[cv2.IMWRITE_JPEG_QUALITY,97])
        if count in [4495,8990]:cv2.imwrite(str(out/f'hq_frame_{count}.jpg'),result,[cv2.IMWRITE_JPEG_QUALITY,97])
        enc.stdin.write(result.tobytes());count+=1
        if count%150==0:print('PROGRESS',count,round(count/fps,1),'video_seconds',round(time.perf_counter()-tick,1),'elapsed',flush=True)
finally:
    enc.stdin.close()
    for r in readers:r.stdout.close();r.wait()
if enc.wait():raise RuntimeError('Encoding failed')
info=dict(source_dimensions=list(source_size),canvas=[w,h],uniform_map_scale=scale,frames=count,requested_seconds=duration,fps='30000/1001',gopro_start=start,dji_start=start+preset.config['offset_dji_minus_gopro'],interpolation='GPU bicubic with composed inverse maps, direct from larger decoded sources' if '--gpu' in sys.argv else 'Lanczos4',encoder='h264_nvenc p6 CQ17',elapsed_seconds=time.perf_counter()-tick)
(out/('sample_info.json' if '--sample' in sys.argv else 'hq_render_info.json')).write_text(json.dumps(info,indent=2))
print('DONE',info,flush=True)
