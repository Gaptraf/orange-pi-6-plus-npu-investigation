"""Numerical NPU tests. Run one case per process; CPU fallback is forbidden."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

ROOT = Path(os.environ.get('OPI6_EXPERIMENT_ROOT', '/opt/opi6-npu-llm/stage1'))
p = argparse.ArgumentParser()
p.add_argument('--case', required=True)
p.add_argument('--runs', type=int, default=10)
p.add_argument('--tag', default='initial')
p.add_argument('--provider', default='ZhouyiExecutionProvider')
p.add_argument('--m', type=int, default=4)
p.add_argument('--k', type=int, default=32)
p.add_argument('--n', type=int, default=16)
p.add_argument('--log-level', type=int, default=2)
p.add_argument('--edge-inputs', action='store_true')
p.add_argument('--pause-ms', type=float, default=0)
p.add_argument('--start-pattern', type=int, default=0)
p.add_argument('--a-scale', type=float, default=1/16)
p.add_argument('--b-scale', type=float, default=1/16)
p.add_argument('--y-scale', type=float, default=1/4)
a = p.parse_args()
work = ROOT/'runs'/f'{a.tag}_{a.case}'
work.mkdir(parents=True, exist_ok=True)
for key, value in {
    'AIPULIB_PATH': '/usr/share/cix/lib/onnxruntime',
    'OPERATOR_PATH': '/usr/share/cix/lib/onnxruntime/operator',
    'GRAPH_PATH': str(work/'graph'),
    'INTERMIDIATE_PATH': str(work/'intermediate'),
}.items():
    os.environ[key] = value
for d in ['graph', 'intermediate']:
    (work/d).mkdir(exist_ok=True)
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort

rng = np.random.default_rng(20261015)
dtype = np.float16 if a.case.endswith('fp16') else np.float32
kind = T.FLOAT16 if dtype == np.float16 else T.FLOAT
M, K, N = a.m, a.k, a.n
quantized = a.case.startswith('qmatmul') or a.case.startswith('qadd')
addition = a.case.startswith('add') or a.case.startswith('qadd')
dynamic = 'dynamic' in a.case or addition
shape_a = [M, K]
shape_b = [M, K] if addition else [K, N]
shape_y = shape_a if addition else [M, N]
initializers = []
if quantized:
    dtype, kind = np.int8, T.INT8
weight = rng.integers(-8, 9, size=shape_b).astype(dtype) if quantized else rng.uniform(-0.5, 0.5, size=shape_b).astype(dtype)
inputs = [h.make_tensor_value_info('A', kind, shape_a)]
if dynamic:
    inputs.append(h.make_tensor_value_info('B', kind, shape_b))
else:
    initializers.append(nh.from_array(weight, 'B'))
if quantized:
    for name, value in [('as', np.float32(a.a_scale)), ('bs', np.float32(a.b_scale)), ('ys', np.float32(a.y_scale)),
                        ('az', np.int8(0)), ('bz', np.int8(0)), ('yz', np.int8(0))]:
        initializers.append(nh.from_array(np.asarray(value), name))
    if addition:
        nodes = [h.make_node('DequantizeLinear', ['A', 'as', 'az'], ['Af']),
                 h.make_node('DequantizeLinear', ['B', 'bs', 'bz'], ['Bf']),
                 h.make_node('Add', ['Af', 'Bf'], ['Yf']),
                 h.make_node('QuantizeLinear', ['Yf', 'ys', 'yz'], ['Y'])]
    else:
        nodes = [h.make_node('QLinearMatMul', ['A', 'as', 'az', 'B', 'bs', 'bz', 'ys', 'yz'], ['Y'])]
else:
    nodes = [h.make_node('Add' if addition else 'MatMul', ['A', 'B'], ['Y'])]
model = h.make_model(h.make_graph(nodes, a.case, inputs, [h.make_tensor_value_info('Y', kind, shape_y)], initializers),
                     opset_imports=[h.make_opsetid('', 13)], ir_version=10)
onnx.checker.check_model(model)
onnx.save(model, work/'model.onnx')
so = ort.SessionOptions()
so.log_severity_level = a.log_level
so.log_verbosity_level = 1
so.enable_profiling = True
so.profile_file_prefix = str(work/'ort_profile')
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
if a.provider != 'CPUExecutionProvider':
    so.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
result = {'case': a.case, 'runs_requested': a.runs, 'provider_requested': a.provider,
          'ort_version': ort.__version__, 'available_providers': ort.get_available_providers(),
          'input_shape': shape_a, 'weight_shape': shape_b, 'dtype': np.dtype(dtype).name,
          'runs_completed': 0, 'numerical_failures': 0, 'max_abs_error': 0.0}
result['edge_inputs'] = a.edge_inputs
result['pause_ms'] = a.pause_ms
result['start_pattern'] = a.start_pattern
result['quantization_scales'] = {'a': float(np.float32(a.a_scale)), 'b': float(np.float32(a.b_scale)), 'y': float(np.float32(a.y_scale))} if quantized else None
(work/'started.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'event': 'create_session', **result}), flush=True)
start = time.perf_counter()
session = ort.InferenceSession(str(work/'model.onnx'), sess_options=so, providers=[a.provider])
session.disable_fallback()
result['session_create_s'] = time.perf_counter()-start
result['session_providers'] = session.get_providers()
print(json.dumps({'event': 'session_ready', 'seconds': result['session_create_s']}), flush=True)
timings = []
for i in range(a.runs):
    x = rng.integers(-8, 9, size=shape_a).astype(dtype) if quantized else rng.uniform(-0.5, 0.5, size=shape_a).astype(dtype)
    b = (rng.integers(-8, 9, size=shape_b).astype(dtype) if quantized else rng.uniform(-0.5, 0.5, size=shape_b).astype(dtype)) if dynamic else weight
    if quantized and a.edge_inputs:
        pattern = (i+a.start_pattern) % 10
        for value in [x, *([b] if dynamic else [])]:
            if pattern == 0:
                value.fill(0)
            elif pattern == 1:
                value.fill(127)
            elif pattern == 2:
                value.fill(-128)
            elif pattern == 3:
                value.flat[::2], value.flat[1::2] = 127, -128
            else:
                value[:] = rng.integers(-128, 128, size=value.shape).astype(dtype)
    feeds = {'A': x, **({'B': b} if dynamic else {})}
    if i == 0:
        np.savez(work/'first_inputs.npz', **feeds, B_reference=b)
    if i and a.pause_ms:
        time.sleep(a.pause_ms/1000)
    start = time.perf_counter()
    actual = session.run(None, feeds)[0]
    timings.append((time.perf_counter()-start)*1000)
    reference = x.astype(np.float64)+b.astype(np.float64) if addition else x.astype(np.float64)@b.astype(np.float64)
    if quantized:
        sa, sb, sy = (float(np.float32(v)) for v in (a.a_scale, a.b_scale, a.y_scale))
        reference = (x.astype(np.float64)*sa+b.astype(np.float64)*sb)/sy if addition else reference*sa*sb/sy
        reference = np.clip(np.rint(reference), -128, 127)
        passed = np.all(np.abs(actual.astype(np.float64)-reference) <= 1)
        result['tolerance'] = 'absolute <= 1 INT8 output unit'
    else:
        atol, rtol = (2e-3, 2e-3) if dtype == np.float16 else (1e-4, 1e-3)
        passed = np.allclose(actual, reference, atol=atol, rtol=rtol)
        result['tolerance'] = {'atol': atol, 'rtol': rtol}
    error = float(np.max(np.abs(actual.astype(np.float64)-reference)))
    result['max_abs_error'] = max(result['max_abs_error'], error)
    result['numerical_failures'] += int(not passed)
    result['runs_completed'] = i+1
    if i == 0:
        np.savez(work/'first_outputs.npz', actual=actual, reference=reference)
    if i < 3 or (i+1) % 100 == 0:
        print(json.dumps({'event': 'inference', 'iteration': i+1, 'ms': timings[-1], 'error': error, 'pass': bool(passed)}), flush=True)
profile = Path(session.end_profiling())
events = json.loads(profile.read_text())
providers = Counter(e.get('args', {}).get('provider') for e in events if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
result['profile_provider_events'] = dict(providers)
result['npu_execution_proven'] = providers.get('ZhouyiExecutionProvider', 0) == a.runs and providers.get('CPUExecutionProvider', 0) == 0
result['first_run_ms'] = timings[0]
steady = timings[1:] or timings
result['warm_ms'] = {'p50': float(np.median(steady)), 'p95': float(np.percentile(steady, 95)), 'mean': float(np.mean(steady))}
result['success'] = result['numerical_failures'] == 0 and (result['npu_execution_proven'] or a.provider == 'CPUExecutionProvider')
(work/'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
raise SystemExit(0 if result['success'] else 2)
