"""Screen Qwen-shaped Transformer operations with strict NPU-only partitioning."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

p=argparse.ArgumentParser()
p.add_argument('--case',required=True)
p.add_argument('--runs',type=int,default=10)
p.add_argument('--provider',default='ZhouyiExecutionProvider')
a=p.parse_args()
root=Path(os.environ.get('OPI6_EXPERIMENT_ROOT', '/opt/opi6-npu-llm/stage2'))
work=root/'ops'/(a.case+('_cpu' if a.provider=='CPUExecutionProvider' else ''))
work.mkdir(parents=True,exist_ok=True)
for name in ['graph','intermediate']: (work/name).mkdir(exist_ok=True)
os.environ.update(AIPULIB_PATH='/usr/share/cix/lib/onnxruntime',OPERATOR_PATH='/usr/share/cix/lib/onnxruntime/operator',
                  GRAPH_PATH=str(work/'graph'),INTERMIDIATE_PATH=str(work/'intermediate'))
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
rng=np.random.default_rng(2042)
nodes=[];initializers=[];inputs=[]
def inp(name,shape,kind=T.FLOAT16):
    inputs.append(h.make_tensor_value_info(name,kind,shape))
def const(name,value,dtype=np.float16):
    initializers.append(nh.from_array(np.asarray(value,dtype=dtype),name))
def node(op,ins,outs,**attrs): nodes.append(h.make_node(op,ins,outs,**attrs))
shape=[1,8,896]
kind=T.FLOAT16
if a.case in ['mul_fp16','silu_fp16','sigmoid_fp16','silu_exp_fp16','rmsnorm_fp16']:
    inp('X',shape)
    if a.case=='mul_fp16':
        inp('B',shape);node('Mul',['X','B'],['Y'])
    elif a.case=='silu_fp16':
        node('Sigmoid',['X'],['S']);node('Mul',['X','S'],['Y'])
    elif a.case=='sigmoid_fp16': node('Sigmoid',['X'],['Y'])
    elif a.case=='silu_exp_fp16':
        inp('ones',shape)
        node('Neg',['X'],['negative']);node('Exp',['negative'],['exp'])
        node('Add',['exp','ones'],['denominator']);node('Div',['X','denominator'],['Y'])
    else:
        const('eps',1e-6); gamma=rng.uniform(0.8,1.2,896).astype(np.float16);const('gamma',gamma)
        node('Mul',['X','X'],['square']);node('ReduceMean',['square'],['mean'],axes=[-1],keepdims=1)
        node('Add',['mean','eps'],['var']);node('Sqrt',['var'],['rms']);node('Div',['X','rms'],['norm']);node('Mul',['norm','gamma'],['Y'])
elif a.case in ['masked_softmax_fp16','masked_softmax_inputs_fp16']:
    shape=[1,14,8,8];inp('X',shape)
    mask=np.where(np.triu(np.ones((8,8)),1)>0,-10000,0).astype(np.float16)
    if a.case=='masked_softmax_inputs_fp16': inp('mask',shape)
    else: const('mask',mask)
    node('Add',['X','mask'],['masked']);node('Softmax',['masked'],['Y'],axis=-1)
elif a.case in ['rope_fp16','rope_inputs_fp16','rope_inputs_mul_fp16']:
    shape=[1,14,8,64];inp('X',shape)
    inv=1/(1e6**(np.arange(0,64,2,dtype=np.float64)/64))
    angle=np.arange(8)[:,None]*inv[None,:]
    angle=np.concatenate([angle,angle],axis=-1)[None,None,:,:]
    cosine=np.cos(angle).astype(np.float16);sine=np.sin(angle).astype(np.float16)
    if a.case in ['rope_inputs_fp16','rope_inputs_mul_fp16']:
        inp('cosine',shape);inp('sine',shape)
    else:
        const('cosine',cosine);const('sine',sine)
    const('split',[32,32],np.int64)
    node('Split',['X','split'],['left','right'],axis=-1)
    if a.case=='rope_inputs_fp16': node('Neg',['right'],['neg'])
    else:
        const('minus',-1);node('Mul',['right','minus'],['neg'])
    node('Concat',['neg','left'],['rotated'],axis=-1);node('Mul',['X','cosine'],['xc']);node('Mul',['rotated','sine'],['xs']);node('Add',['xc','xs'],['Y'])
elif a.case=='kv_concat_fp16':
    shape=[1,2,8,64];inp('X',[1,2,7,64]);inp('B',[1,2,1,64]);node('Concat',['X','B'],['Y'],axis=2)
elif a.case=='qattention_bmm_int8':
    shape=[1,14,8,8];kind=T.INT8
    inp('X',[1,14,8,64],T.INT8);inp('B',[1,14,64,8],T.INT8)
    for key,value in [('sx',0.05),('sb',0.05),('sy',0.1)]: const(key,value,np.float32)
    const('zero',0,np.int8)
    node('QLinearMatMul',['X','sx','zero','B','sb','zero','sy','zero'],['Y'])
else: raise ValueError(a.case)
model=h.make_model(h.make_graph(nodes,a.case,inputs,[h.make_tensor_value_info('Y',kind,shape)],initializers),opset_imports=[h.make_opsetid('',13)],ir_version=10)
onnx.checker.check_model(model);onnx.save(model,work/'model.onnx')
so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1;so.log_severity_level=2
so.enable_profiling=True;so.profile_file_prefix=str(work/'profile')
if a.provider!='CPUExecutionProvider': so.add_session_config_entry('session.disable_cpu_ep_fallback','1')
result={'case':a.case,'runs':0,'numerical_failures':0,'max_abs_error':0.0}
print(json.dumps({'event':'session_start','case':a.case}),flush=True)
start=time.perf_counter()
session=ort.InferenceSession(str(work/'model.onnx'),sess_options=so,providers=[a.provider]);session.disable_fallback()
result['compile_seconds']=time.perf_counter()-start
times=[]
for i in range(a.runs):
    feeds={}
    for meta in session.get_inputs():
        feeds[meta.name]=rng.integers(-16,17,size=meta.shape,dtype=np.int8) if kind==T.INT8 else rng.normal(0,0.5,meta.shape).astype(np.float16)
    if a.case=='masked_softmax_inputs_fp16': feeds['mask']=np.broadcast_to(mask,shape).copy()
    elif a.case in ['rope_inputs_fp16','rope_inputs_mul_fp16']:
        feeds['cosine']=np.broadcast_to(cosine,shape).copy();feeds['sine']=np.broadcast_to(sine,shape).copy()
    elif a.case=='silu_exp_fp16': feeds['ones']=np.ones(shape,dtype=np.float16)
    x=feeds['X'].astype(np.float64)
    if a.case=='mul_fp16': gold=x*feeds['B'].astype(np.float64)
    elif a.case=='sigmoid_fp16': gold=1/(1+np.exp(-x))
    elif a.case in ['silu_fp16','silu_exp_fp16']: gold=x/(1+np.exp(-x))
    elif a.case=='rmsnorm_fp16': gold=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+float(np.float16(1e-6)))*gamma.astype(np.float64)
    elif a.case in ['masked_softmax_fp16','masked_softmax_inputs_fp16']:
        z=x+mask.astype(np.float64);z=z-z.max(axis=-1,keepdims=True);gold=np.exp(z)/np.exp(z).sum(axis=-1,keepdims=True)
    elif a.case in ['rope_fp16','rope_inputs_fp16','rope_inputs_mul_fp16']: gold=x*cosine.astype(np.float64)+np.concatenate([-x[...,32:],x[...,:32]],axis=-1)*sine.astype(np.float64)
    elif a.case=='kv_concat_fp16': gold=np.concatenate([x,feeds['B'].astype(np.float64)],axis=2)
    else: gold=np.clip(np.rint((x@feeds['B'].astype(np.float64))*float(np.float32(0.05))**2/float(np.float32(0.1))),-128,127)
    start=time.perf_counter();actual=session.run(None,feeds)[0];times.append((time.perf_counter()-start)*1000)
    error=float(np.max(np.abs(actual.astype(np.float64)-gold)))
    good=bool(error<=1) if kind==T.INT8 else bool(np.allclose(actual,gold,atol=0.003,rtol=0.003))
    result['runs']+=1;result['numerical_failures']+=int(not good);result['max_abs_error']=max(result['max_abs_error'],error)
    if i==0: np.savez(work/'sample.npz',**feeds,actual=actual,reference=gold)
events=json.loads(Path(session.end_profiling()).read_text())
providers=Counter(e.get('args',{}).get('provider') for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider'))
result['profile_providers']=dict(providers)
result['npu_only']=providers.get('ZhouyiExecutionProvider',0)>0 and providers.get('CPUExecutionProvider',0)==0
result['p50_ms']=float(np.median(times[1:]));result['p95_ms']=float(np.percentile(times[1:],95))
result['provider_requested']=a.provider
result['success']=(result['npu_only'] or a.provider=='CPUExecutionProvider') and result['numerical_failures']==0
(work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
raise SystemExit(0 if result['success'] else 2)
