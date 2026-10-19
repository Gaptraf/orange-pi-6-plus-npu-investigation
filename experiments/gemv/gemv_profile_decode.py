import sys,tarfile,json,contextlib,statistics
from pathlib import Path
sys.path=[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
root=Path(__file__).resolve().parent.parent;src=root/'host-sdk/gemv-captures';src.mkdir(exist_ok=True)
with tarfile.open(root/'artifacts/gemv-profile-captures.tar.gz') as t:
    for m in t.getmembers():assert m.isfile() and '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
    t.extractall(src)
from AIPUBuilder.Profiler.main import main
dest=root/'artifacts/gemv-profile-decoded';dest.mkdir(exist_ok=True);summary={}
for name in ['full-prefetch','full-prefetch3']:
    folder=src/(name+'-profile');rows=[]
    for f in sorted((folder/'capture').glob('run*.bin')):
        target=dest/(name+'-'+f.stem+'.json')
        sys.argv=['cixprofiler',str(folder/'graph-1200mhz.json'),str(f),'--method','json','-f','1200','-o',str(target)]
        with target.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):main()
        data=json.loads(target.read_text())['profile_data'];sg=data['subgraphs'];nodes=[n for s in sg for n in s['nodes']]
        rows.append({'run':f.stem,'core_ids':sorted(set(s['Core ID'] for s in sg)),
                     'nodes':[{'name':n['Layer Name'],'core':n['core_id'],'start':n['start'],'end':n['end']} for n in nodes],
                     'span_us':(max(n['end'] for n in nodes)-min(n['start'] for n in nodes))/1200,
                     'read_bytes':data['AIPU Read Bytes'],'constant_read_bytes':data['Constant Read Bytes']})
    summary[name]={'runs':rows,'cores':sorted(set(c for r in rows for c in r['core_ids'])),'median_span_us':statistics.median(r['span_us'] for r in rows)}
(root/'artifacts/gemv-profile-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
