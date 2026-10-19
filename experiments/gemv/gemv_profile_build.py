"""Compile separate profile-only binaries; measured speed binaries stay untouched."""
import sys,os,shutil
from pathlib import Path
sys.path=[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
root=Path(__file__).resolve().parent.parent
base=Path('/home/researcher/opi6plusnpu-host-20261016/gemv')
import AIPUBuilder.core
sys.path.insert(0,str(base/'full-prefetch'))
import aipubt_gemv
from AIPUBuilder._C._tools import aipugb
for name in ['full-prefetch','full-prefetch3']:
    src=base/name;work=base/(name+'-profile');work.mkdir(exist_ok=True);os.chdir(work)
    rc=aipugb(['aipugb',str(src/'graph.def'),'-w',str(src/'weight.bin'),'--target','X2_1204MP3','--gm_size','4096','-L',str(src),'-p','--fast_perf','-o',str(work/'profile.bin')])
    if rc:raise SystemExit(rc)
    dest=root/'host-sdk/gemv'/(name+'-profile');dest.mkdir(exist_ok=True)
    for f in work.rglob('*'):
        if f.is_file():
            rel=f.relative_to(work);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target);print(str(rel),f.stat().st_size,flush=True)
