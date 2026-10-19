"""Pack unchanged projection weights and compile a private custom-op graph."""
import argparse,os,sys,subprocess,shutil,json
from pathlib import Path
sys.path=[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--tiny',action='store_true');p.add_argument('--variant',default='prefetch',choices=['direct','wire','constants','dma_constants','dma','prefetch','prefetch3']);a=p.parse_args()
root=Path(__file__).resolve().parent.parent
name=('tiny' if a.tiny else 'full')+'-'+a.variant
work=Path('/home/researcher/opi6plusnpu-host-20261016/gemv')/name;work.mkdir(parents=True,exist_ok=True)
dest=root/'host-sdk/gemv'/name;dest.mkdir(parents=True,exist_ok=True)
os.chdir(work)
os.environ['AIPUPLUGIN_PATH']=str(work)
shutil.copyfile(root/'scripts/aipubt_gemv.py',work/'aipubt_gemv.py')
src=root/'scripts/gemv_packed.cl'
if a.variant!='direct':src=root/f'scripts/gemv_{"prefetch" if a.variant=="prefetch3" else a.variant}.cl'
shutil.copyfile(src,work/'gemv_packed.cl')
cc='/home/researcher/opi6plusnpu-host-20261016/sdk/AIPUBuilder/_C/bin/aipuocc'
for kind in ['-c','-S']:
    r=subprocess.run([cc,'-target','aipu','-mcpu=X2_1204','-O2',kind,'gemv_packed.cl','-o','gemv_packed'+('.o' if kind=='-c' else '.s')],capture_output=True,text=True)
    (work/f'compile{kind}.log').write_text(r.stdout+r.stderr);print(r.stdout,r.stderr,flush=True);r.check_returncode()
raw=np.fromfile(root/'host-sdk/compiler-inputs/weight.bin',dtype=np.int8)
w0=raw[19456:19456+4358144].reshape(4864,896)
w1=raw[4397056:4397056+4358144].reshape(4864,896)
assert not raw[:19456].any() and not raw[4377600:4397056].any()
k,n=(32,64) if a.tiny else (896,4864)
if a.tiny and a.variant=='prefetch':n=256  # exercise ping-pong reuse on each TEC
w0=w0[:n,:k].copy();w1=w1[:n,:k].copy()
def pack(w):return w.reshape(n//32,4,8,k//4,4).transpose(0,3,1,2,4).copy()
packed=np.concatenate([pack(w0).ravel(),pack(w1).ravel()]);packed.tofile(work/'weight.bin')
ir=f'''model_name=gemv
layer_number=2
data_format=NHWC
precision=float32
batch_size=1
compat_quantized_model=false
input_tensors=[x]
output_tensors=[yg,yu]

layer_id=0
layer_name=input
layer_type=Input
layer_bottom=[]
layer_bottom_shape=[]
layer_bottom_type=[]
layer_top=[x]
layer_top_shape=[[1,{k}]]
layer_top_type=[int8]

layer_id=1
layer_name=gemv
layer_type=Opi6PackedGemv
layer_bottom=[x]
layer_bottom_shape=[[1,{k}]]
layer_bottom_type=[int8]
layer_top=[yg,yu]
layer_top_shape=[[1,{n}],[1,{n}]]
layer_top_type=[int8,int8]
wg_type=int8
wg_shape=[{n*k}]
wg_offset=0
wg_size={n*k}
wu_type=int8
wu_shape=[{n*k}]
wu_offset={n*k}
wu_size={n*k}
k4={k//4}
n32={n//32}
'''
parts=[1600,1600,1664] if a.variant=='prefetch3' else [n]
if a.variant=='prefetch3':
    assert not a.tiny
    header=ir.split('\nlayer_id=1')[0].replace('layer_number=2','layer_number=4').replace('output_tensors=[yg,yu]','output_tensors=[yg0,yu0,yg1,yu1,yg2,yu2]')
    template=ir.split('\nlayer_id=1')[1]
    chunks=[];offset=0;start=0
    for z,rows in enumerate(parts):
        for w in [w0,w1]:chunks.append(w[start:start+rows].reshape(rows//32,4,8,k//4,4).transpose(0,3,1,2,4).copy().ravel())
        node=template.replace('layer_name=gemv',f'layer_name=gemv{z}').replace('layer_top=[yg,yu]',f'layer_top=[yg{z},yu{z}]')
        node=node.replace(f'[[1,{n}],[1,{n}]]',f'[[1,{rows}],[1,{rows}]]').replace(f'[{n*k}]',f'[{rows*k}]')
        node=node.replace(f'wg_offset=0',f'wg_offset={offset}').replace(f'wu_offset={n*k}',f'wu_offset={offset+rows*k}')
        node=node.replace(f'size={n*k}',f'size={rows*k}').replace(f'n32={n//32}',f'n32={rows//32}')
        header+=f'\nlayer_id={z+1}'+node;start+=rows;offset+=2*rows*k
    packed=np.concatenate(chunks);packed.tofile(work/'weight.bin');ir=header
(work/'graph.def').write_text(ir)
rng=np.random.default_rng(20261016)
for c in range(8):
    x=rng.integers(-16,17,k,dtype=np.int8)
    x.tofile(work/f'input{c}.bin')
    for j,(w,scale) in enumerate([(w0,4772),(w1,4252)]):
        accum=w.astype(np.int64)@x.astype(np.int64)
        y=np.clip((accum*scale+262144)>>19,-128,127).astype(np.int8)
        if a.variant=='wire': y=np.tile(x[:32],n//32) if j==0 else np.full(n,17,dtype=np.int8)
        if a.variant in ['constants','dma_constants']: y=pack(w).ravel()[:n]
        start=0
        for z,rows in enumerate(parts):
            y[start:start+rows].tofile(work/f'expected{c}_{2*z+j}.bin');start+=rows
import AIPUBuilder.core
sys.path.insert(0,str(work))
import aipubt_gemv
from AIPUBuilder._C._tools import aipugb
rc=aipugb(['aipugb','graph.def','-w','weight.bin','--target','X2_1204MP3','--gm_size','4096','-L',str(work),'-o','graph0.bin'])
if rc:raise SystemExit(rc)
for f in work.iterdir():
    if f.is_file():shutil.copyfile(f,dest/f.name)
print(json.dumps({'name':name,'k':k,'n':n,'weight_bytes':packed.nbytes,'binary_bytes':(work/'graph0.bin').stat().st_size,'destination':str(dest)}),flush=True)
