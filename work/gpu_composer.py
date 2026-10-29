"""OpenCL Lanczos sampling and frozen-preset color blending."""
import numpy as np
import pyopencl as cl

KERNEL=r'''
float smooth5(float x){x=clamp(x,0.0f,1.0f);return x*x*x*(x*(x*6-15)+10);}
float sinc4(float x){if(fabs(x)<1e-5f)return 1; if(fabs(x)>=4)return 0;return native_sin(3.14159265359f*x)*native_sin(.785398163397f*x)/(2.46740110027f*x*x);}
float cubic(float x){x=fabs(x);if(x<=1)return 1.25f*x*x*x-2.25f*x*x+1; if(x<2)return -.75f*x*x*x+3.75f*x*x-6*x+3;return 0;}
float3 sample8(__global const uchar *im,int sw,int sh,float x,float y){
  if(x < -2 || y < -2 || x > sw+1 || y > sh+1)return (float3)(0);
  int ix=(int)floor(x),iy=(int)floor(y);float wx[4],wy[4],sx=0,sy=0;
  for(int i=0;i<4;i++){wx[i]=cubic(x-(ix+i-1));wy[i]=cubic(y-(iy+i-1));sx+=wx[i];sy+=wy[i];}
  float3 total=(float3)(0);
  for(int j=0;j<4;j++){
    int yy=iy+j-1;if(yy<0||yy>=sh)continue;
    for(int i=0;i<4;i++){int xx=ix+i-1;if(xx<0||xx>=sw)continue;total+=convert_float3(vload3(yy*sw+xx,im))*(wx[i]*wy[j]);}
  }
  return clamp(total/(sx*sy),(float3)(0),(float3)(255));
}
float3 correct(float3 raw,float3 gain){
  float light=(raw.x+raw.y+raw.z)/3;
  float neutral=fmin(fmin(raw.x,raw.y),raw.z)/fmax(fmax(fmax(raw.x,raw.y),raw.z),1.0f);
  float protect=smooth5((light-110)/100)*smooth5((neutral-.45f)/.35f);
  return mix(raw*gain,raw,protect);
}
__kernel void compose(__global const uchar *a,__global const uchar *b,
 __global const float *mx0,__global const float *my0,__global const float *mx1,__global const float *my1,
 __global const float *wa,__global const float *wb,__global const float *g,
 __global uchar *out,int count,int sw,int sh){
  int n=get_global_id(0);if(n>=count)return;
  float3 sum=(float3)(0);
  if(wa[n]>0){float3 raw=sample8(a,sw,sh,mx0[n],my0[n]);sum+=correct(raw,vload3(0,g))*wa[n];}
  if(wb[n]>0){float3 raw=sample8(b,sw,sh,mx1[n],my1[n]);sum+=correct(raw,vload3(1,g))*wb[n];}
  vstore3(convert_uchar3_sat_rtz(sum),n,out);
}
'''

class GpuComposer:
    def __init__(self,maps,weights,gains,half,shape):
        devices=[d for p in cl.get_platforms() for d in p.get_devices(device_type=cl.device_type.GPU)]
        devices.sort(key=lambda d:('NVIDIA' not in d.vendor,-d.global_mem_size))
        if not devices:raise RuntimeError('No OpenCL GPU')
        self.ctx=cl.Context([devices[0]]);self.queue=cl.CommandQueue(self.ctx)
        self.kernel=cl.Program(self.ctx,KERNEL).build().compose
        self.name=devices[0].name;self.shape=shape
        mf=cl.mem_flags
        self.inputs=[cl.Buffer(self.ctx,mf.READ_ONLY,2704*1520*3) for _ in range(2)]
        self.output=cl.Buffer(self.ctx,mf.WRITE_ONLY,int(np.prod(shape)))
        self.host=np.empty(shape,np.uint8)
        def upload(a):return cl.Buffer(self.ctx,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=np.ascontiguousarray(a,np.float32))
        self.buffers=[upload(a) for pair in maps for a in pair]+[upload(w[:,:,0]) for w in weights]
        self.gains=upload(np.stack([gains[0]/half,gains[1]*half]))

    def compose(self,frames):
        for buf,im in zip(self.inputs,frames):cl.enqueue_copy(self.queue,buf,im,is_blocking=False)
        n=self.shape[0]*self.shape[1]
        self.kernel(self.queue,((n+255)//256*256,),(256,),*self.inputs,*self.buffers,self.gains,self.output,np.int32(n),np.int32(2704),np.int32(1520))
        cl.enqueue_copy(self.queue,self.host,self.output,is_blocking=True)
        return self.host
