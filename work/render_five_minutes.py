import sys,time,json,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base/'outputs/field_merge/adjusted/saved_preset'))
import cv2,numpy as np
from numba import njit,prange,set_num_threads
from scipy.signal import correlate
from apply_preset import FrozenStitchPreset
cv2.setNumThreads(4);set_num_threads(4)
preset=FrozenStitchPreset();d=preset.data
out=base/'outputs/field_merge/five_minutes';out.mkdir(exist_ok=True)
paths=[Path(s['path']) for s in preset.config['sources']]
start=85.;duration=300.;offset=preset.config['offset_dji_minus_gopro'];fps=30000/1001
def run(args):return subprocess.check_output(list(map(str,args)))
def audio(p,t,n):return np.frombuffer(run(['ffmpeg','-v','error','-ss',t,'-i',p,'-t',n,'-vn','-ac','1','-ar','8000','-f','f32le','-']),np.float32).copy()
def frame(p,t):return cv2.imdecode(np.frombuffer(run(['ffmpeg','-v','error','-noautorotate','-ss',t,'-i',p,'-map','0:v:0','-frames:v','1','-vf','scale=1600:900','-f','image2pipe','-vcodec','png','-']),np.uint8),1)
@njit(parallel=True)
def colorblend(a,b,g0,g1,wa,wb):
    result=np.empty(a.shape,np.uint8)
    for y in prange(a.shape[0]):
        for x in range(a.shape[1]):
            for index in range(2):
                c=a if index==0 else b
                g=g0 if index==0 else g1
                weight=wa[y,x,0] if index==0 else wb[y,x,0]
                r0=c[y,x,0];r1=c[y,x,1];r2=c[y,x,2]
                brightness=(r0+r1+r2)/3
                neutrality=min(r0,r1,r2)/max(max(r0,r1,r2),1)
                u=min(1.,max(0.,(brightness-110)/100));v=min(1.,max(0.,(neutrality-.45)/.35))
                protection=(u*u*u*(u*(u*6-15)+10))*(v*v*v*(v*(v*6-15)+10))
                for channel in range(3):
                    value=(c[y,x,channel]*g[channel]*(1-protection)+c[y,x,channel]*protection)*weight
                    if index==0:result[y,x,channel]=np.uint8(min(255,max(0,value)))
                    else:result[y,x,channel]=np.uint8(min(255,max(0,float(result[y,x,channel])+value)))
    return result
# Blend float first, so quantization happens only after both source contributions.
@njit(parallel=True)
def fused(a,b,gains,half,wa,wb):
    result=np.empty(a.shape,np.uint8)
    for y in prange(a.shape[0]):
        for x in range(a.shape[1]):
            total=np.zeros(3,np.float32)
            for i in range(2):
                c=a if i==0 else b
                r0=c[y,x,0];r1=c[y,x,1];r2=c[y,x,2]
                brightness=(r0+r1+r2)/3.
                neutrality=min(r0,r1,r2)/max(max(r0,r1,r2),1.)
                u=min(1.,max(0.,(brightness-110)/100));v=min(1.,max(0.,(neutrality-.45)/.35))
                protect=(u*u*u*(u*(u*6-15)+10))*(v*v*v*(v*(v*6-15)+10))
                wt=wa[y,x,0] if i==0 else wb[y,x,0]
                for k in range(3):
                    corrected=np.float32(c[y,x,k]*gains[i,k])
                    corrected=np.float32(corrected*half[k] if i==1 else corrected/half[k])
                    total[k]+=np.float32((corrected*(1-protect)+c[y,x,k]*protect)*wt)
            for k in range(3):result[y,x,k]=np.uint8(min(255,max(0,total[k])))
    return result
def fast(frames):
    cs=[]
    for i,im in enumerate(frames):
        y0,y1,x0,x1=map(int,d[f'placement_{i}']);c=np.zeros((preset.height,preset.width,3),np.float32)
        c[y0:y1,x0:x1]=cv2.remap(im,d[f'map_x_{i}'],d[f'map_y_{i}'],cv2.INTER_LINEAR).astype(np.float32)
        if i==1:c=cv2.remap(c,d['rx'],d['ry'],cv2.INTER_LINEAR)
        cs.append(c)
    return fused(cs[0],cs[1],d['gains'],d['half_gain'],d['weight_0'],d['weight_1'])
if '--check' in sys.argv:
    checks=[]
    for t in [90,235,380]:
        a=audio(paths[0],t+offset-.5,9);b=audio(paths[1],t,8)
        a-=a.mean();b-=b.mean();corr=correlate(a,b,mode='valid',method='fft');pos=int(np.argmax(corr));seg=a[pos:pos+len(b)]
        score=float(np.dot(seg,b)/np.sqrt(np.dot(seg,seg)*np.dot(b,b)))
        measured=offset-.5+pos/8000
        ims=[frame(paths[0],t+offset),frame(paths[1],t)]
        expected=preset.compose(*ims);actual=fast(ims)
        diff=np.abs(expected.astype(np.int16)-actual.astype(np.int16))
        check=dict(gopro_time=t,offset=measured,correlation=score,fast_mean_pixel_difference=float(diff.mean()),fast_max_pixel_difference=int(diff.max()))
        checks.append(check);print('CHECK',check,flush=True)
        assert diff.max()<=1
        assert abs(measured-offset)<.5/fps and score>.2,'Synchronization requires adjustment'
        cv2.imwrite(str(out/f'preview_{t}.jpg'),actual)
    tick=time.perf_counter()
    for _ in range(10):fast(ims)
    print('COMPOSE_FPS',10/(time.perf_counter()-tick),flush=True)
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
if '--render' in sys.argv:
    readers=[subprocess.Popen(list(map(str,['ffmpeg','-v','error','-noautorotate','-ss',t,'-i',p,'-t',duration,'-map','0:v:0','-vf','scale=1600:900','-pix_fmt','bgr24','-f','rawvideo','-'])),stdout=subprocess.PIPE) for p,t in zip(paths,[start+offset,start])]
    dest=out/'soccer_panorama_5min.mp4'
    enc=subprocess.Popen(list(map(str,['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{preset.width}x{preset.height}','-r','30000/1001','-i','-','-ss',start,'-i',paths[1],'-map','0:v:0','-map','1:a:0','-t',duration,'-c:v','h264_nvenc','-preset','p5','-cq','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart',dest])),stdin=subprocess.PIPE)
    count=0;tick=time.perf_counter()
    try:
        while count<round(duration*fps):
            chunks=[r.stdout.read(1600*900*3) for r in readers]
            if any(len(c)!=1600*900*3 for c in chunks):raise RuntimeError('Unexpected source end')
            enc.stdin.write(fast([np.frombuffer(c,np.uint8).reshape(900,1600,3) for c in chunks]).tobytes())
            count+=1
            if count%600==0:print('PROGRESS',count,round(count/fps,1),'video_seconds',round(time.perf_counter()-tick,1),'elapsed',flush=True)
    finally:
        enc.stdin.close()
        for r in readers:r.stdout.close();r.wait()
    if enc.wait():raise RuntimeError('Encoding failed')
    (out/'render_info.json').write_text(json.dumps(dict(gopro_start=start,dji_start=start+offset,duration_requested=duration,frames=count,fps='30000/1001',canvas=[preset.width,preset.height],preset=str(base/'outputs/field_merge/adjusted/saved_preset'),elapsed_seconds=time.perf_counter()-tick),indent=2))
    print('DONE',count,flush=True)
