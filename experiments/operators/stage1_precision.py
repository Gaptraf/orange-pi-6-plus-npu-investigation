"""Check whether the observed FLOAT32 Add agrees with FP16 intermediates."""
import json
from pathlib import Path
import numpy as np
root=Path('/opt/opi6-npu-llm/stage1')
work=root/'runs/support_add_fp32'
inputs=np.load(work/'first_inputs.npz')
actual=np.load(work/'first_outputs.npz')['actual']
a,b=inputs['A'],inputs['B']
variants={'float32':a+b,
          'round_output_to_fp16':(a+b).astype(np.float16).astype(np.float32),
          'fp16_inputs_and_output':(a.astype(np.float16)+b.astype(np.float16)).astype(np.float32)}
result={key:{'exact_match':bool(np.array_equal(actual,value)),
             'max_abs_error':float(np.max(np.abs(actual.astype(np.float64)-value.astype(np.float64))))} for key,value in variants.items()}
(root/'precision.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
