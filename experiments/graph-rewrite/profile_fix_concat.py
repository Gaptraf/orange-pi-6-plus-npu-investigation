"""Narrow IR workaround: preserve byte concatenation of identically quantized FCs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent.parent;src=root/'host-sdk/split-inputs'
text=(src/'graph.def').read_text();blocks=text.strip().split('\n\n');scales={};fixes=[]
for idx,block in enumerate(blocks):
    fields=dict(line.split('=',1) for line in block.splitlines() if '=' in line)
    if 'layer_type' not in fields:continue
    if fields['layer_type']=='Concat':
        names=fields['layer_bottom'].strip('[]').split(',');incoming=[scales[n] for n in names]
        assert len(incoming)==3 and len(set(incoming))==1 and incoming[0]>1
        assert fields['layer_top_type']=='[int8]' and fields['layer_top_zp']=='[0]'
        fixes.append({'layer':fields['layer_name'],'previous_output_scale':fields['layer_top_scale'],'correct_scale':incoming[0],'operation':'identity requantization for byte concatenation'})
        fields['layer_top_scale']=str([incoming[0]])
        fields['scale_value']='[1,1,1]';fields['shift_value']='[0,0,0]'
        blocks[idx]='\n'.join(k+'='+v for k,v in fields.items())
    if 'layer_top' in fields and 'layer_top_scale' in fields:
        names=fields['layer_top'].strip('[]').split(',');values=json.loads(fields['layer_top_scale']);assert len(names)==len(values)
        scales.update(zip(names,values))
assert len(fixes)==2
(src/'graph-fixed.def').write_text('\n\n'.join(blocks)+'\n')
(root/'artifacts/profile-concat-fix.json').write_text(json.dumps(fixes,indent=2));print(json.dumps(fixes,indent=2))
