"""Isolate split-FC arithmetic from NPU concatenation; preserve quantization."""
import gc,json,os,time
from pathlib import Path
root=Path('/opt/opi6-npu-llm/profile-20261016');work=root/'split-no-concat';work.mkdir(exist_ok=True);os.chdir(work)
for n in ['graph','intermediate']:(work/n).mkdir(exist_ok=True)
os.environ.update(AIPULIB_PATH='/usr/share/cix/lib/onnxruntime',OPERATOR_PATH='/usr/share/cix/lib/onnxruntime/operator',GRAPH_PATH=str(work/'graph'),INTERMIDIATE_PATH=str(work/'intermediate'),ENABLE_PROFILE='off',DUMP_FILE='on',TILING_METHOD='fps',TOOLKIT_LOG_LEVEL='0')
os.sched_setaffinity(0,{0})
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as h
m=onnx.load('/opt/opi6-npu-llm/review-20261016/split-gate-up/model.onnx')
nodes=[n for n in m.graph.node if n.op_type!='Concat'];del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.output[:]
for i,n in enumerate(nodes):m.graph.output.append(h.make_tensor_value_info(n.output[0],onnx.TensorProto.INT8,[1,[1600,1600,1664][i%3]]))
onnx.checker.check_model(m);onnx.save(m,work/'model.onnx')
def session(path,provider):
    o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1
    if provider!='CPUExecutionProvider':o.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    s=ort.InferenceSession(str(path),sess_options=o,providers=[provider]);s.disable_fallback();return s
s=session(work/'model.onnx','ZhouyiExecutionProvider');cpu=session('/opt/opi6-npu-llm/performance/npu_fused_m1/group2.onnx','CPUExecutionProvider')
ref=Path('/opt/opi6-npu-llm/review-20261016/probes/npu1_m1_fps_g2');errors=[];xs=[]
for c in range(8):
    x=np.fromfile(ref/f'input{c}.bin',dtype=np.int8).reshape(1,896);xs.append(x);x.tofile(work/f'input{c}.bin')
    ys=s.run(None,{'X':x});refs=cpu.run(None,{'X':x})
    for j,y in enumerate(ys):y.tofile(work/f'expected{c}_{j}.bin')
    for j in range(2):
        y=np.concatenate(ys[j*3:j*3+3],axis=1);errors.append(int(np.abs(y.astype(np.int16)-refs[j].astype(np.int16)).max()))
result={'max_error_lsb_vs_original_cpu':max(errors),'errors':errors,'valid':max(errors)<=1,'outputs':[o.name for o in s.get_outputs()]}
if result['valid']:
    for i in range(20):s.run(None,{'X':xs[i%8]})
    times=[]
    for i in range(240):
        t=time.perf_counter_ns();ys=s.run(None,{'X':xs[i%8]});ys=[np.concatenate(ys[j*3:j*3+3],axis=1) for j in range(2)];times.append((time.perf_counter_ns()-t)/1000)
    result.update(ort_with_cpu_concat_us=times,median_us=float(np.median(times)))
(work/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='ort_with_cpu_concat_us'}));del s;gc.collect()
