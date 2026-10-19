"""One bounded test of explicit N-axis partitioning, preserving quantization."""
import gc,json,os,time
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as h,numpy_helper as nh
root=Path('/opt/opi6-npu-llm/review-20261016');work=root/'split-gate-up';work.mkdir(exist_ok=True)
for n in ['graph','intermediate']:(work/n).mkdir(exist_ok=True)
os.environ.update(AIPULIB_PATH='/usr/share/cix/lib/onnxruntime',OPERATOR_PATH='/usr/share/cix/lib/onnxruntime/operator',GRAPH_PATH=str(work/'graph'),INTERMIDIATE_PATH=str(work/'intermediate'),ENABLE_PROFILE='off',DUMP_FILE='on',TILING_METHOD='fps',TOOLKIT_LOG_LEVEL='0')
os.sched_setaffinity(0,{0})
m=onnx.load('/opt/opi6-npu-llm/performance/npu_fused_m1/group2.onnx');original=list(m.graph.node);ws={x.name:nh.to_array(x) for x in m.graph.initializer}
newnodes=[];newinit=[];cuts=[0,1600,3200,4864]
for i,node in enumerate(original):
    assert node.op_type=='QLinearMatMul'
    wname=node.input[3];assert ws[wname].shape==(896,4864)
    parts=[]
    for j,(a,b) in enumerate(zip(cuts,cuts[1:])):
        name=f'split{i}_{j}';newinit.append(nh.from_array(ws[wname][:,a:b].copy(),name+'_W'))
        inputs=list(node.input);inputs[3]=name+'_W';parts.append(name+'_Y')
        newnodes.append(h.make_node('QLinearMatMul',inputs,[parts[-1]],name=name))
    newnodes.append(h.make_node('Concat',parts,list(node.output),axis=1))
del m.graph.node[:];m.graph.node.extend(newnodes)
keep=[x for x in m.graph.initializer if not x.name.endswith('_W')]
del m.graph.initializer[:];m.graph.initializer.extend(keep+newinit);onnx.checker.check_model(m);onnx.save(m,work/'model.onnx')
opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;opts.add_session_config_entry('session.disable_cpu_ep_fallback','1')
s=ort.InferenceSession(str(work/'model.onnx'),sess_options=opts,providers=['ZhouyiExecutionProvider']);s.disable_fallback()
base=root/'probes/npu1_m1_fps_g2';xs=[];errs=[]
for c in range(8):
    x=np.fromfile(base/f'input{c}.bin',dtype=np.int8).reshape(1,896);xs.append(x);x.tofile(work/f'input{c}.bin')
    for j,y in enumerate(s.run(None,{'X':x})):
        ref=np.fromfile(base/f'expected{c}_{j}.bin',dtype=np.int8).reshape(y.shape)
        err=int(np.abs(y.astype(np.int16)-ref.astype(np.int16)).max());errs.append(err)
        y.tofile(work/f'expected{c}_{j}.bin')
if max(errs)>1:
    result={'parts_per_projection':3,'output_lengths':[1600,1600,1664],'max_difference_from_original_lsb':max(errs),'errors_per_case_output':errs,'valid':False,'timing_skipped':True}
    (work/'ort.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0)
for i in range(20):s.run(None,{'X':xs[i%8]})
times=[];gc.disable()
for i in range(240):
    t=time.perf_counter_ns();s.run(None,{'X':xs[i%8]});times.append((time.perf_counter_ns()-t)/1000)
gc.enable();result={'parts_per_projection':3,'output_lengths':[1600,1600,1664],'max_difference_from_original_lsb':max(errs),'median_us':float(np.median(times)),'p95_us':float(np.percentile(times,95)),'raw_us':times}
(work/'ort.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));del s;gc.collect()
