"""Use the old NPU baseline's exact inputs and quantify output differences."""
import sys,json,tarfile,itertools
from pathlib import Path
sys.path=[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
import numpy as np
root=Path(__file__).resolve().parent.parent
variant=sys.argv[1]
dest=root/'host-sdk/gemv'/variant
raw=np.fromfile(root/'host-sdk/compiler-inputs/weight.bin',dtype=np.int8)
weights=[raw[19456:19456+4358144].reshape(4864,896).astype(np.int64),raw[4397056:4397056+4358144].reshape(4864,896).astype(np.int64)]
errors=[]
with tarfile.open(root/'artifacts/gemv-baseline-cases.tar.gz') as t:
    for c in range(8):
        data=t.extractfile(f'input{c}.bin').read();(dest/f'input{c}.bin').write_bytes(data)
        x=np.frombuffer(data,dtype=np.int8).astype(np.int64)
        custom=[]
        for j,scale in enumerate([4772,4252]):
            y=np.clip((weights[j]@x*scale+262144)>>19,-128,127).astype(np.int8)
            custom.append(y.astype(np.int16))
            start=0
            for z,rows in enumerate([1600,1600,1664] if variant=='full-prefetch3' else [4864]):
                y[start:start+rows].tofile(dest/f'expected{c}_{2*z+j}.bin');start+=rows
        old=[np.frombuffer(t.extractfile(f'expected{c}_{j}.bin').read(),dtype=np.int8).astype(np.int16) for j in range(2)]
        if c==0:mapping=min(itertools.permutations(range(2)),key=lambda m:max(np.max(np.abs(custom[j]-old[m[j]])) for j in range(2)))
        errors.append([{'max_abs_lsb':int(np.max(np.abs(custom[j]-old[mapping[j]]))), 'different':int(np.count_nonzero(custom[j]!=old[mapping[j]]))} for j in range(2)])
result={'custom_to_baseline_outputs':list(mapping),'errors':errors,'same_inputs':True}
(root/f'artifacts/gemv-{variant}-accuracy.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
