import sys,json
from pathlib import Path
sys.path.insert(0,r'C:\Users\Tony Openclaw\Documents\Codex\2026-10-21\task-build-and-test-a-dual\work\packages')
import cv2,numpy as np
base=Path(__file__).resolve().parents[1];out=base/'outputs'
frames=[cv2.imread(str(base/'work'/n)) for n in ['current_frame.png','frame100.png','frame200.png']]
im=np.median(np.stack(frames),axis=0).astype(np.uint8);h,w=im.shape[:2]
xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
def smooth(t):
 t=np.clip(t,0,1);return t*t*t*(t*(t*6-15)+10)
def trace(image,anchors,kind,radius=24):
 pts=[]
 for x in range(int(anchors[0][0]*w),int(anchors[-1][0]*w),4):
  y=np.interp(x/w,np.array(anchors)[:,0],np.array(anchors)[:,1])*h
  lo=max(0,int(y-radius));hi=min(h,int(y+radius));b,g,r=image[lo:hi,x].astype(float).T
  if kind=='yellow':good=(r>g*.99)&(g>b*1.5)&(r>110)&(g>90)
  else:good=(np.minimum(b,r)>g*.84)&(g>125)&(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<65)
  ids=np.where(good)[0]
  if len(ids):pts.append((x,float(np.median(ids)+lo)))
 p=np.array(pts);assert len(p)>40
 old=np.interp(np.arange(w),p[:,0],p[:,1]);old=cv2.GaussianBlur(old.reshape(1,-1),(0,0),18).ravel()
 return p,old
anchors=[[.04,.392],[.10,.49],[.19,.617],[.27,.715],[.36,.795],[.43,.837],[.50,.858],[.59,.882],[.66,.891],[.70,.867],[.77,.75],[.85,.603],[.93,.445],[.965,.362]]
p,old=trace(im,anchors,'yellow',40)
# A single smooth fit now includes both junctions rather than ending before them.
coef=np.polyfit(p[:,0]/w,p[:,1]/h,4);dest=np.polyval(coef,np.arange(w)/w)*h
fade=np.exp(-((yy-dest[None,:])/(h*.085))**2)
edge=smooth((xx/w-.025)/.065)*smooth((.93-xx/w)/.06)
dy=np.clip((old-dest)[None,:],-55,55)*fade*edge
mx=xx.copy();my=yy+dy.astype(np.float32)
center=.434*w
seam=np.exp(-((xx-center)/(w*.01))**2)*np.exp(-((yy-dest[None,:])/(h*.025))**4)
mx+=(np.where(xx<center,-1,1)*w*.0055*seam).astype(np.float32)
my+=((np.interp(mx.ravel(),np.arange(w),old).reshape(h,w)-old[None,:])*fade*edge).astype(np.float32)
checks=[]
# Independent, gently faded corrections for the three inner marking transitions.
for name,anc,kind,degree in [
 ('inner_white',[[.37,.554],[.45,.582],[.53,.59],[.61,.588],[.69,.568]],'white',3),
 ('inner_yellow',[[.37,.513],[.45,.527],[.53,.53],[.61,.523],[.69,.505]],'yellow',3),
 ('far_yellow',[[.42,.379],[.48,.383],[.54,.378],[.60,.370]],'yellow',2)]:
 current=cv2.remap(im,mx,my,cv2.INTER_LINEAR)
 p,line=trace(current,anc,kind,18)
 coef2=np.polyfit(p[:,0]/w,p[:,1]/h,degree);goal=np.polyval(coef2,np.arange(w)/w)*h
 correction=np.clip(line-goal,-10,10)
 env=smooth((xx/w-anc[0][0])/.045)*smooth((anc[-1][0]-xx/w)/.045)
 delta=(correction[None,:]*env*np.exp(-((yy-goal[None,:])/(h*.022))**2)).astype(np.float32)
 # Compose with the current map, so each pass still samples the original image.
 mx=cv2.remap(mx,xx,yy+delta,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
 my=cv2.remap(my,xx,yy+delta,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
 checks.append({'region':name,'maximum_adjustment':float(abs(delta).max())})
cv2.imwrite(str(base/'work/map_v2_x.pgm'),np.clip(np.rint(mx),0,65535).astype(np.uint16))
cv2.imwrite(str(base/'work/map_v2_y.pgm'),np.clip(np.rint(my),0,65535).astype(np.uint16))
preview=cv2.remap(frames[0],mx,my,cv2.INTER_LINEAR)
cv2.imwrite(str(out/'curve_v2_preview.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,96])
(out/'curve_v2_settings.json').write_text(json.dumps({'inner_corrections':checks,'foreground_maximum':float(abs(dy).max()),'method':'Fixed smooth geometric remap, extended across the foreground junctions; real source pixels only'},indent=2))
print(checks)

