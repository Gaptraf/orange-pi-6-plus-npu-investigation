"""Prepare genuine Qwen layer weights and an independently checked FP32 reference."""
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

root=Path('/opt/opi6-npu-llm/stage2');folder=root/'qwen-layer0'
manifest=json.loads((folder/'manifest.json').read_text())
header=json.loads((folder/'safetensors_header.json').read_text())
embedding=folder/'embedding-first512.bin'
meta=header['model.embed_tokens.weight']
assert meta['dtype']=='BF16'
count=512*manifest['config']['hidden_size']*2
offset=8+(folder/'safetensors_header.json').stat().st_size+meta['data_offsets'][0]
if not embedding.exists():
    url=f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/model.safetensors?stage2_embedding=first512"
    request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{offset+count-1}','User-Agent':'opi6-npu-lab'})
    with urllib.request.urlopen(request,timeout=60) as response:
        assert response.status==206
        data=response.read(count+1);assert len(data)==count
    embedding.write_bytes(data)
def decode(path,shape,dtype='BF16'):
    raw=path.read_bytes()
    if dtype=='BF16': array=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4')
    elif dtype=='F32': array=np.frombuffer(raw,dtype='<f4')
    else: raise ValueError(dtype)
    return array.reshape(shape).copy()
weights={}
for name,meta in manifest['tensors'].items():
    path=folder/meta['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    weights[name.removeprefix('model.layers.0.')]=decode(path,meta['shape'],meta['dtype'])
np.savez(root/'weights_fp32.npz',**weights)
table=decode(embedding,[512,896])
torch.set_num_threads(1)
config=Qwen2Config(**manifest['config']);config._attn_implementation='eager'
layer=Qwen2DecoderLayer(config,0).float().eval()
layer.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()},strict=True)
rng=np.random.default_rng(71425)
calibration=table[rng.integers(0,512,size=(12,8))]
evaluation=table[rng.integers(0,512,size=(8,8))]
outputs=[]
with torch.no_grad():
    positions=torch.arange(8).unsqueeze(0)
    mask=torch.triu(torch.full((8,8),torch.finfo(torch.float32).min),diagonal=1)[None,None]
    for sample in evaluation:
        y=layer(torch.from_numpy(sample[None]),attention_mask=mask,position_ids=positions)[0]
        outputs.append(y[0].numpy())
np.savez(root/'reference_data.npz',calibration=calibration,evaluation=evaluation,torch_output=np.stack(outputs))
report={'model_revision':manifest['revision'],'torch_version':torch.__version__,'dtype':'float32',
        'real_layer_weights':True,'embedding_rows':512,'calibration_sequences':12,'evaluation_sequences':8,
        'sequence_length':8,'input_selection':'seeded sequences of token IDs 0..511, not natural-language evaluation',
        'embedding_sha256':hashlib.sha256(embedding.read_bytes()).hexdigest()}
(root/'reference_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
