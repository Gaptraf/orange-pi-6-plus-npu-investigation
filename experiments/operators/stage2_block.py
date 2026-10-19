"""One genuine Qwen block: explicit CPU operations and seven INT8 NPU projections.

Reference: the saved output of the unmodified Transformers Qwen2DecoderLayer.
Calibration and evaluation sequences are separate. This is not text generation.
"""
from collections import Counter
import json
import os
from pathlib import Path
import time

root=Path('/opt/opi6-npu-llm/stage2');work=root/'hybrid_block';work.mkdir(exist_ok=True)
for name in ['graph','intermediate']: (work/name).mkdir(exist_ok=True)
os.environ.update(AIPULIB_PATH='/usr/share/cix/lib/onnxruntime',OPERATOR_PATH='/usr/share/cix/lib/onnxruntime/operator',
                  GRAPH_PATH=str(work/'graph'),INTERMIDIATE_PATH=str(work/'intermediate'))
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
import onnxruntime as ort
weights=dict(np.load(root/'weights_fp32.npz'))
data=np.load(root/'reference_data.npz')
projections=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
stats={name:{'xmax':0.0,'ymax':0.0} for name in projections}

def rms(x,w): return (x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+np.float32(1e-6)))*w
def rope(x):
    inv=np.float32(1)/(np.float32(1e6)**(np.arange(0,64,2,dtype=np.float32)/np.float32(64)))
    angle=np.arange(x.shape[1],dtype=np.float32)[:,None]*inv[None,:]
    angle=np.concatenate([angle,angle],axis=-1)[None]
    rotated=np.concatenate([-x[...,32:],x[...,:32]],axis=-1)
    return x*np.cos(angle)+rotated*np.sin(angle)
def forward(x,linear):
    normalized=rms(x,weights['input_layernorm.weight'])
    q=linear('self_attn.q_proj',normalized).reshape(-1,14,64).transpose(1,0,2)
    k=linear('self_attn.k_proj',normalized).reshape(-1,2,64).transpose(1,0,2)
    v=linear('self_attn.v_proj',normalized).reshape(-1,2,64).transpose(1,0,2)
    q,k=rope(q),rope(k)
    k=np.repeat(k,7,axis=0);v=np.repeat(v,7,axis=0)
    scores=(q@k.transpose(0,2,1))/np.float32(8)
    mask=np.triu(np.full((len(x),len(x)),np.finfo(np.float32).min,dtype=np.float32),1)
    scores=scores+mask;scores=scores-np.max(scores,axis=-1,keepdims=True)
    probabilities=np.exp(scores);probabilities=probabilities/np.sum(probabilities,axis=-1,keepdims=True)
    attention=(probabilities@v).transpose(1,0,2).reshape(len(x),896)
    residual=x+linear('self_attn.o_proj',attention)
    normalized=rms(residual,weights['post_attention_layernorm.weight'])
    gate=linear('mlp.gate_proj',normalized);up=linear('mlp.up_proj',normalized)
    activated=gate/(np.float32(1)+np.exp(-gate))*up
    return residual+linear('mlp.down_proj',activated)
def fp32_linear(name,x,collect=False):
    y=x@weights[name+'.weight'].T
    if collect:
        stats[name]['xmax']=max(stats[name]['xmax'],float(np.max(np.abs(x))))
        stats[name]['ymax']=max(stats[name]['ymax'],float(np.max(np.abs(y))))
    if name+'.bias' in weights: y=y+weights[name+'.bias']
    return y

for sample in data['calibration']: forward(sample,lambda name,x:fp32_linear(name,x,True))
cpu_fp32=np.stack([forward(sample,fp32_linear) for sample in data['evaluation']])
torch_expected=data['torch_output']
reference_error=float(np.max(np.abs(cpu_fp32-torch_expected)))
assert np.allclose(cpu_fp32,torch_expected,atol=1e-4,rtol=1e-4),reference_error
print(json.dumps({'event':'independent_reference_matches_transformers','max_error':reference_error}),flush=True)

parameters={};sessions={};compilation={}
for name in projections:
    folder=work/name;folder.mkdir(exist_ok=True)
    weight=weights[name+'.weight'].T.copy()
    sx=np.float32(max(stats[name]['xmax']*1.1/127,1e-8))
    sw=np.float32(max(float(np.max(np.abs(weight)))/127,1e-8))
    sy=np.float32(max(stats[name]['ymax']*1.1/127,1e-8))
    wq=np.clip(np.rint(weight/sw),-127,127).astype(np.int8)
    parameters[name]=(sx,sw,sy,wq)
    initializers=[nh.from_array(wq,'W'),nh.from_array(sx,'sx'),nh.from_array(sw,'sw'),nh.from_array(sy,'sy'),nh.from_array(np.int8(0),'z')]
    node=h.make_node('QLinearMatMul',['X','sx','z','W','sw','z','sy','z'],['Y'])
    model=h.make_model(h.make_graph([node],name,[h.make_tensor_value_info('X',T.INT8,[8,weight.shape[0]])],
                                   [h.make_tensor_value_info('Y',T.INT8,[8,weight.shape[1]])],initializers),opset_imports=[h.make_opsetid('',13)],ir_version=10)
    onnx.checker.check_model(model);onnx.save(model,folder/'model.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.log_severity_level=2
    options.enable_profiling=True;options.profile_file_prefix=str(folder/'profile')
    options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    start=time.perf_counter()
    session=ort.InferenceSession(str(folder/'model.onnx'),sess_options=options,providers=['ZhouyiExecutionProvider']);session.disable_fallback()
    sessions[name]=session;compilation[name]=time.perf_counter()-start
    print(json.dumps({'event':'projection_compiled','name':name,'seconds':compilation[name]}),flush=True)

node_error={name:0 for name in projections};node_counts=Counter();saturation=Counter();total_values=Counter();npu_times=[]
def quant_linear(name,x,npu):
    sx,sw,sy,wq=parameters[name]
    rounded=np.rint(x/sx)
    saturation[name]+=int(np.count_nonzero((rounded<-128)|(rounded>127)));total_values[name]+=rounded.size
    xq=np.clip(rounded,-128,127).astype(np.int8)
    # Exact integer arithmetic is outside the measured NPU call.
    integer=xq.astype(np.int32)@wq.astype(np.int32)
    expected=np.clip(np.rint(integer.astype(np.float64)*float(sx)*float(sw)/float(sy)),-128,127).astype(np.int8)
    if npu:
        start=time.perf_counter();actual=sessions[name].run(None,{'X':xq})[0];npu_times.append((time.perf_counter()-start)*1000)
        error=int(np.max(np.abs(actual.astype(np.int16)-expected.astype(np.int16))))
        node_error[name]=max(node_error[name],error);node_counts[name]+=1
        y=actual.astype(np.float32)*sy
    else: y=expected.astype(np.float32)*sy
    if name+'.bias' in weights: y=y+weights[name+'.bias']
    return y

cpu_quant=np.stack([forward(sample,lambda name,x:quant_linear(name,x,False)) for sample in data['evaluation']])
npu_output=np.stack([forward(sample,lambda name,x:quant_linear(name,x,True)) for sample in data['evaluation']])
def metrics(actual,expected):
    actual=actual.astype(np.float64);expected=expected.astype(np.float64)
    return {'max_abs_error':float(np.max(np.abs(actual-expected))),
            'relative_l2_error':float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
            'cosine_similarity':float(np.sum(actual*expected)/(np.linalg.norm(actual)*np.linalg.norm(expected)))}
providers=Counter()
for session in sessions.values():
    events=json.loads(Path(session.end_profiling()).read_text())
    providers.update(e.get('args',{}).get('provider') for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider'))
report={'reference_matches_transformers':True,'reference_max_abs_error':reference_error,
        'sequence_length':8,'evaluation_sequences':8,'calibration_sequences':12,
        'npu_projection_calls':dict(node_counts),'profile_providers':dict(providers),'projection_max_int8_error':node_error,
        'npu_vs_fp32':metrics(npu_output,torch_expected),'cpu_quant_vs_fp32':metrics(cpu_quant,torch_expected),
        'npu_vs_cpu_quant':metrics(npu_output,cpu_quant),'compilation_seconds':compilation,
        'npu_call_p50_ms':float(np.median(npu_times)),
        'input_saturation_fraction':{name:saturation[name]/total_values[name] for name in projections},
        'cpu_operations':['RMSNorm','RoPE','attention QK/PV','causal mask and softmax','bias and residual','SiLU','quantization/dequantization'],
        'npu_operations':projections,'all_npu_calls_within_1_lsb':all(value<=1 for value in node_error.values()),
        'note':'Synthetic token sequences from real embedding rows 0..511; not language quality or speed benchmark'}
report['numerical_prototype_valid']=report['all_npu_calls_within_1_lsb'] and providers.get('ZhouyiExecutionProvider')==56 and providers.get('CPUExecutionProvider',0)==0
(work/'result.json').write_text(json.dumps(report,indent=2));np.savez(work/'outputs.npz',npu=npu_output,cpu_quant=cpu_quant,torch_fp32=torch_expected)
(work/'quantization.json').write_text(json.dumps({name:{'sx':float(x[0]),'sw':float(x[1]),'sy':float(x[2])} for name,x in parameters.items()},indent=2))
print(json.dumps(report,indent=2),flush=True)
raise SystemExit(0 if report['numerical_prototype_valid'] else 2)
