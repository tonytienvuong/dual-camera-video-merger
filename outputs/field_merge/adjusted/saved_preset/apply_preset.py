"""Apply frozen accepted geometry. Requires NumPy and OpenCV.

No calibration or feature detection is repeated. Inputs must be synchronized
DJI and GoPro BGR frames, decoded without autorotation, in that order.
"""
from pathlib import Path
import json
import cv2
import numpy as np

def smoothstep(t):
    t=np.clip(t,0,1)
    return t*t*t*(t*(t*6-15)+10)

class FrozenStitchPreset:
    def __init__(self, folder=None):
        folder=Path(folder) if folder else Path(__file__).resolve().parent
        self.config=json.loads((folder/'preset.json').read_text())
        with np.load(folder/self.config['maps']) as data:
            self.data={name:data[name] for name in data.files}
        self.width,self.height=self.config['output']['canvas']

    def compose(self,dji_frame,gopro_frame):
        result=np.zeros((self.height,self.width,3),np.float32)
        d=self.data
        for i,frame in enumerate([dji_frame,gopro_frame]):
            frame=cv2.resize(frame,(1600,900))
            y0,y1,x0,x1=map(int,d[f'placement_{i}'])
            c=np.zeros_like(result)
            c[y0:y1,x0:x1]=cv2.remap(frame,d[f'map_x_{i}'],d[f'map_y_{i}'],cv2.INTER_LINEAR).astype(np.float32)
            if i==1:c=cv2.remap(c,d['rx'],d['ry'],cv2.INTER_LINEAR)
            original=c.copy()
            c*=d['gains'][i]
            if i==1:c*=d['half_gain']
            else:c/=d['half_gain']
            brightness=original.mean(axis=2)
            neutrality=original.min(axis=2)/np.maximum(original.max(axis=2),1)
            protect=(smoothstep((brightness-110)/100)*smoothstep((neutrality-.45)/.35))[:,:,None]
            c=c*(1-protect)+original*protect
            result+=c*d[f'weight_{i}']
        return np.clip(result,0,255).astype(np.uint8)
