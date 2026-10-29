"""Resumable full HQ stitch with per-chapter audio clock-drift fitting."""
import sys,json,time,subprocess,math,os
from pathlib import Path
base=Path(__file__).resolve().parents[1]
plan_path=base/'outputs/field_merge/full_match/timeline_probe.json'
timeline=json.loads(plan_path.read_text())
# Only initialize the approved HQ map and GPU composer, not its test renderer.
saved_argv=sys.argv[:];sys.argv=['render_hq.py','--gpu']
exec(compile((Path(__file__).parent/'render_hq.py').read_text().split('paths=[Path')[0],str(Path(__file__).parent/'render_hq.py'),'exec'))
sys.argv=saved_argv
root=Path(__import__('os').environ.get('SOCCER_SOURCE_DIR', r'F:\Video\test'));out=base/'outputs/field_merge/full_match';out.mkdir(exist_ok=True)
parts=base/'work/full_match_parts';parts.mkdir(exist_ok=True)
fps=30000/1001;start=50.;end=3450.;total_frames=round((end-start)*fps)
models={}
for di in range(2):
    for go in timeline['go']:
        anchors=[a for a in timeline['anchors'] if a['dji_index']==di and a['go_name']==go['name']]
        if not anchors:continue
        assert len(anchors)>=2
        gg=np.array([a['go_global'] for a in anchors]);tt=np.array([a['dji_time'] for a in anchors])
        slope,intercept=np.polyfit(gg,tt,1)
        residual=np.abs(gg*slope+intercept-tt)
        assert residual.max()<.012 and abs(slope-1)<.0002
        models[(di,go['name'])]=dict(slope=float(slope),intercept=float(intercept),max_anchor_error_seconds=float(residual.max()))
di1_model=models[(1,'GP030499.MP4')]
switch=-di1_model['intercept']/di1_model['slope']
first_end=(timeline['dji'][0]['duration']-models[(0,'GP030499.MP4')]['intercept'])/models[(0,'GP030499.MP4')]['slope']
cuts=sorted(set([start,end,switch]+[s['global_start'] for s in timeline['go'] if start<s['global_start']<end]))
segments=[]
for n,(left,right) in enumerate(zip(cuts,cuts[1:])):
    f0=max(0,int(math.ceil((left-start)*fps-1e-6)));f1=min(total_frames,int(math.ceil((right-start)*fps-1e-6)))
    gstart=start+f0/fps;count=f1-f0
    go=next(s for s in timeline['go'] if s['global_start']<=gstart<s['global_start']+s['duration'])
    di=0 if gstart<switch else 1;model=models[(di,go['name'])]
    ds=model['intercept']+model['slope']*gstart
    segments.append(dict(index=n,global_start=gstart,global_stop=start+f1/fps,frames=count,start_frame=f0,end_frame=f1,
        go_name=go['name'],go_local_start=gstart-go['global_start'],dji_name=timeline['dji'][di]['name'],dji_local_start=ds,
        dji_index=di,clock_model=model,file=str(parts/f'part_{n:02d}.mkv')))
plan=dict(start_gopro=start,end_gopro=end,frames=total_frames,fps='30000/1001',duration_seconds=total_frames/fps,
    canvas=[w,h],dji_switch_gopro_time=switch,dji_boundary_gap_seconds=switch-first_end,
    boundary_policy='Hold last DJI frame for the approximately two-frame recording gap; GoPro timeline and audio continue',
    trimmed='GoPro first 50 s and footage after 3450 s: missing paired coverage, setup or camera pickup',segments=segments)
(out/'render_plan.json').write_text(json.dumps(plan,indent=2))
print('PLAN',plan['duration_seconds'],'seconds',len(segments),'segments','DJI_GAP',switch-first_end,flush=True)
if '--plan-only' in sys.argv:sys.exit()
begin=time.perf_counter();completed=0
for seg in segments:
    target=Path(seg['file']);count=seg['frames'];duration=count/fps
    if target.exists() and target.with_suffix('.done.json').exists():
        info=json.loads(target.with_suffix('.done.json').read_text());assert info['segment']==seg
        completed+=count;print('REUSE',seg['index'],completed,flush=True);continue
    if target.exists():target.rename(target.with_name(target.stem+f'_incomplete_{int(time.time())}.mkv'))
    readers=[]
    for i,(name,t) in enumerate([(seg['dji_name'],seg['dji_local_start']),(seg['go_name'],seg['go_local_start'])]):
        slope=seg['clock_model']['slope'] if i==0 else 1
        filt=f'setpts=(PTS-STARTPTS)/{slope:.12f},fps=30000/1001:start_time=0,scale=2704:1520:flags=lanczos,tpad=stop_mode=clone:stop_duration=0.20'
        readers.append(subprocess.Popen(list(map(str,['ffmpeg','-v','error','-noautorotate','-ss',max(0,t),'-i',root/name,'-frames:v',count,'-map','0:v:0','-vf',filt,'-pix_fmt','bgr24','-f','rawvideo','-'])),stdout=subprocess.PIPE))
    enc=subprocess.Popen(list(map(str,['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','30000/1001','-i','-','-an','-c:v','h264_nvenc','-preset','p6','-cq','17','-pix_fmt','yuv420p',target])),stdin=subprocess.PIPE)
    print('SEGMENT_START',seg['index'],seg['global_start'],seg['global_stop'],flush=True)
    last_update=time.perf_counter();local=0
    try:
        while local<count:
            if (base/'work/stop_full_match.flag').exists():raise RuntimeError('Stop requested via control file')
            chunks=[r.stdout.read(2704*1520*3) for r in readers]
            if any(len(c)!=2704*1520*3 for c in chunks):raise RuntimeError('Unexpected decoder end')
            ims=[np.frombuffer(c,np.uint8).reshape(1520,2704,3) for c in chunks]
            result=compose(ims)
            if local in [0,count//2,count-1]:cv2.imwrite(str(out/f'segment_{seg["index"]:02d}_frame_{local}.jpg'),result,[cv2.IMWRITE_JPEG_QUALITY,95])
            enc.stdin.write(result.tobytes());local+=1
            if time.perf_counter()-last_update>=20:
                progress=dict(segment=seg['index'],segment_frame=local,completed_frames=completed+local,total_frames=total_frames,
                    percent=100*(completed+local)/total_frames,elapsed_seconds=time.perf_counter()-begin)
                (out/'progress.json').write_text(json.dumps(progress,indent=2));print('PROGRESS',json.dumps(progress),flush=True);last_update=time.perf_counter()
    finally:
        enc.stdin.close()
        for r in readers:r.stdout.close();r.wait()
    assert enc.wait()==0,'Encoder failed'
    raw=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_packets','-select_streams','v:0','-show_entries','stream=nb_read_packets,width,height','-of','json',str(target)]))
    stream=raw['streams'][0];assert int(stream['nb_read_packets'])==count and stream['width']==w and stream['height']==h
    target.with_suffix('.done.json').write_text(json.dumps(dict(segment=seg,probe=raw),indent=2))
    completed+=count;print('SEGMENT_DONE',seg['index'],completed,flush=True)
# Build camera audio timeline with video-based chapter durations. Resampling
# reconciles the few-ms AAC packet overlap at file boundaries.
go_list=parts/'gopro_audio.ffconcat'
go_list.write_text('ffconcat version 1.0\n'+''.join(f"file '{(root/s['name']).as_posix()}'\nduration {s['duration']:.9f}\n" for s in timeline['go']))
audio=parts/'gopro_audio.m4a'
if not audio.exists():
    subprocess.run(['ffmpeg','-v','error','-n','-f','concat','-safe','0','-i',str(go_list),'-map','0:a:0','-vn','-af','aresample=async=1:first_pts=0','-c:a','aac','-b:a','192k',str(audio)],check=True)
video_list=parts/'video.ffconcat'
video_list.write_text('ffconcat version 1.0\n'+''.join(f"file '{Path(s['file']).as_posix()}'\nduration {s['frames']/fps:.9f}\n" for s in segments))
final=out/'final_panorama.mp4'
if final.exists():raise RuntimeError('Final output already exists; refusing overwrite')
print('FINAL_MUX',flush=True)
subprocess.run(list(map(str,['ffmpeg','-v','error','-n','-f','concat','-safe','0','-i',video_list,'-ss',start,'-i',audio,'-map','0:v:0','-map','1:a:0','-t',total_frames/fps,'-c','copy','-movflags','+faststart',final])),check=True)
print('FINAL_DECODE_CHECK',flush=True)
subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(final),'-f','null','-'],check=True)
raw=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(final)]))
(out/'final_probe.json').write_text(json.dumps(raw,indent=2))
v=next(s for s in raw['streams'] if s['codec_type']=='video');assert int(v['nb_frames'])==total_frames
summary=dict(status='complete',frames=total_frames,video_duration=float(v['duration']),container_duration=float(raw['format']['duration']),bytes=int(raw['format']['size']),elapsed_seconds=time.perf_counter()-begin,full_decode='passed')
(out/'completion.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps(summary),flush=True)
