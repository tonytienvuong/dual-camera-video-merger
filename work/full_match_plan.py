import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np
from scipy.signal import correlate
base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base/'outputs/field_merge/adjusted/saved_preset'))
from apply_preset import FrozenStitchPreset
root=Path(__import__('os').environ.get('SOCCER_SOURCE_DIR', r'F:\Video\test'));out=base/'outputs/field_merge/full_match';out.mkdir(exist_ok=True)
go_names=['GOPR0499.MP4','GP010499.MP4','GP020499.MP4','GP030499.MP4','GP040499.MP4']
dji_names=['DJI_20261021203051_0002_D.MP4','DJI_20261021211107_0003_D.MP4']
def run(args):return subprocess.check_output(list(map(str,args)))
def probe(name):
    raw=json.loads(run(['ffprobe','-v','quiet','-show_streams','-show_format','-of','json',root/name]));v=next(s for s in raw['streams'] if s['codec_type']=='video')
    return dict(name=name,duration=float(v['duration']),frames=int(v['nb_frames']),probe=raw)
go=[probe(n) for n in go_names];dji=[probe(n) for n in dji_names]
cursor=0
for s in go:s['global_start']=cursor;cursor+=s['duration']
def go_source(g):return next(s for s in go if s['global_start']<=g<s['global_start']+s['duration'])
def audio(name,t,n):
    x=np.frombuffer(run(['ffmpeg','-v','error','-ss',max(0,t),'-i',root/name,'-t',n,'-vn','-ac','1','-ar','8000','-f','f32le','-']),np.float32).copy();return x-x.mean()
anchors=[]
for index,stamp in [(0,t) for t in [40,380,690,730,1050,1390,1440,1750,2100,2160,2400]]+[(1,t) for t in [2470,2700,2820,2860,3150,3400,3470]]:
    gs=go_source(stamp);approx=stamp-35.59225 if index==0 else stamp-(dji[0]['duration']+35.59225)
    radius=.8 if index==0 else 4
    lo=max(0,approx-radius)
    a=audio(dji[index]['name'],lo,8+2*radius);b=audio(gs['name'],stamp-gs['global_start'],8)
    c=correlate(a,b,mode='valid',method='fft');pos=int(np.argmax(c));actual=lo+pos/8000;seg=a[pos:pos+len(b)]
    score=float(np.dot(seg,b)/np.sqrt(np.dot(seg,seg)*np.dot(b,b)))
    anchor=dict(dji_index=index,go_name=gs['name'],go_global=stamp,go_local=stamp-gs['global_start'],dji_time=actual,correlation=score)
    anchors.append(anchor);print('ANCHOR',anchor,flush=True)
    (out/'timeline_probe.json').write_text(json.dumps(dict(go=go,dji=dji,anchors=anchors),indent=2))
    assert score>.22,'Low-confidence timing anchor'
preset=FrozenStitchPreset()
for stamp in [40,100,700,1400,2100,2440,2470,2800,3400,3470]:
    index=0 if stamp<2450 else 1
    aa=[a for a in anchors if a['dji_index']==index]
    dt=float(np.interp(stamp,[a['go_global'] for a in aa],[a['dji_time'] for a in aa]))
    if stamp<aa[0]['go_global']:dt=aa[0]['dji_time']+(stamp-aa[0]['go_global'])
    if stamp>aa[-1]['go_global']:dt=aa[-1]['dji_time']+(stamp-aa[-1]['go_global'])
    gs=go_source(stamp);ims=[]
    for name,t in [(dji[index]['name'],dt),(gs['name'],stamp-gs['global_start'])]:
        im=cv2.imdecode(np.frombuffer(run(['ffmpeg','-v','error','-noautorotate','-ss',t,'-i',root/name,'-map','0:v:0','-frames:v','1','-vf','scale=1600:900','-f','image2pipe','-vcodec','png','-']),np.uint8),1);ims.append(im)
    cv2.imwrite(str(out/f'preflight_{stamp}.jpg'),preset.compose(*ims))
print('PREFLIGHT_DONE',flush=True)
