"""SiLU workaround: two explicit NPU sessions, with an intermediate host transfer."""
from collections import Counter
import json
import os
from pathlib import Path
import time
root=Path('/opt/opi6-npu-llm/stage2/silu_split');root.mkdir(exist_ok=True)
for name in ['graph','intermediate']: (root/name).mkdir(exist_ok=True)
os.environ.update(AIPULIB_PATH='/usr/share/cix/lib/onnxruntime',OPERATOR_PATH='/usr/share/cix/lib/onnxruntime/operator',
                  GRAPH_PATH=str(root/'graph'),INTERMIDIATE_PATH=str(root/'intermediate'))
import numpy as np
import onnx
from onnx import helper as h,TensorProto as T
import onnxruntime as ort
shape=[1,8,4864]
sessions=[]
for op,names in [('Sigmoid',['X']),('Mul',['X','S'])]:
    model=h.make_model(h.make_graph([h.make_node(op,names,['Y'])],op,
        [h.make_tensor_value_info(name,T.FLOAT16,shape) for name in names],
        [h.make_tensor_value_info('Y',T.FLOAT16,shape)]),opset_imports=[h.make_opsetid('',13)],ir_version=10)
    onnx.checker.check_model(model);onnx.save(model,root/(op+'.onnx'))
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.log_severity_level=2
    options.enable_profiling=True;options.profile_file_prefix=str(root/op)
    options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    s=ort.InferenceSession(str(root/(op+'.onnx')),sess_options=options,providers=['ZhouyiExecutionProvider']);s.disable_fallback()
    sessions.append(s)
rng=np.random.default_rng(2049);failures=0;max_error=0;timings=[]
for i in range(10):
    x=rng.uniform(-10,10,shape).astype(np.float16)
    start=time.perf_counter()
    sig=sessions[0].run(None,{'X':x})[0]
    actual=sessions[1].run(None,{'X':x,'S':sig})[0]
    timings.append((time.perf_counter()-start)*1000)
    expected=x.astype(np.float64)/(1+np.exp(-x.astype(np.float64)))
    max_error=max(max_error,float(np.max(np.abs(actual-expected))))
    failures+=int(not np.allclose(actual,expected,atol=.003,rtol=.003))
    if i==0: np.savez(root/'sample.npz',X=x,actual=actual,reference=expected)
providers=Counter()
for s in sessions:
    events=json.loads(Path(s.end_profiling()).read_text())
    providers.update(e.get('args',{}).get('provider') for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider'))
result={'shape':shape,'input_range':[-10,10],'runs':10,'npu_calls':20,'max_abs_error':max_error,'numerical_failures':failures,
    'profile_providers':dict(providers),'p50_ms':float(np.median(timings[1:])),
    'note':'Two NPU graphs; sigmoid intermediate returns through host memory; not fused SiLU',
    'success':failures==0 and providers.get('ZhouyiExecutionProvider')==20 and providers.get('CPUExecutionProvider',0)==0}
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
raise SystemExit(0 if result['success'] else 2)
