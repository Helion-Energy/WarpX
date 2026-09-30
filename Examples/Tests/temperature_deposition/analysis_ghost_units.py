# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Native ghost-unit fixture runner and independent two-particle oracle."""
from pathlib import Path
import hashlib,itertools,json,math,re
import numpy as np
BOX=re.compile(r'^\(\(([-0-9,]+)\) \(([-0-9,]+)\) \(([-0-9,]+)\)\)$')
FAB=re.compile(rb'\(\(([-0-9,]+)\) \(([-0-9,]+)\) \(([-0-9,]+)\)\) (\d+)\s*$')
def ints(v):return tuple(map(int,v.split(b',' if isinstance(v,bytes) else ',')))
def mf(p):
    lines=Path(str(p)+'_H').read_text().splitlines()
    boxes=[tuple(ints(v) for v in m.groups()) for line in lines if (m:=BOX.match(line))]
    files=[x.split()[1:] for x in lines if x.startswith('FabOnDisk:')];assert len(boxes)==len(files)
    out={}
    for box,(name,off) in zip(boxes,files):
        with (p.parent/name).open('rb') as f:
            f.seek(int(off));line=f.readline();m=FAB.search(line);assert m,line
            lo,hi,typ=map(ints,m.groups()[:3]);nc=int(m.group(4));shape=tuple(h-l+1 for l,h in zip(lo,hi));data=f.read(nc*math.prod(shape)*8)
            assert len(data)==nc*math.prod(shape)*8
        a=np.frombuffer(data,dtype='<f8').reshape((nc,*shape[::-1]));sl=tuple(slice(l-fl,h-fl+1) for l,h,fl in zip(box[0],box[1],lo))[::-1]
        out[box]={'lo':lo,'hi':hi,'a':a,'valid':a[(slice(None),*sl)],'data':data}
    return out

def state_equal(a,b,names):
    all_bytes=True;valid_bytes=True;finite=True;diff=0.
    for name in names:
        x=mf(a/name);y=mf(b/name);assert x.keys()==y.keys()
        for k in x:
            xx,yy=x[k],y[k];assert xx['lo']==yy['lo'] and xx['hi']==yy['hi']
            all_bytes &=xx['data']==yy['data'];valid_bytes &=xx['valid'].tobytes()==yy['valid'].tobytes()
            finite &=bool(np.isfinite(xx['a']).all() and np.isfinite(yy['a']).all())
            if finite:diff=max(diff,float(np.max(abs(xx['valid']-yy['valid']))))
    return dict(exact_all_fabs=all_bytes,exact_valid=valid_bytes,all_fabs_finite=finite,max_valid_difference_K=diff)

def particle_oracle(row):
    c=row['case'];root=Path(row['cwd']);nd=len(c['periodic']);p=c['passes'] if c['filter'] else 0
    shifts=range(-p,p+1);kernel={i:math.comb(2*p,i+p)/4**p for i in shifts}
    results=[]
    # Independent two-particle identity: Var=(v1-v2)^2/2, regardless of the1:9 weights and common nonuniform cubic shape.
    # Physical constants are pinned CODATA values used in the input parser.
    mp=1.67262192595e-27;kb=1.380649e-23
    for name,pos in c['positions'].items():
        for component in range(3):
            amplitude=2*(component+1)**2*mp/kb
            for box,fab in mf(root/f'{name}_T{component}').items():
                lo,hi,typ=box
                coords=np.meshgrid(*[np.arange(l,h+1,dtype=float)+.5*(1-n) for l,h,n in zip(lo,hi,typ)][::-1],indexing='ij')[::-1]
                expect=np.zeros_like(coords[0])
                for delta in itertools.product(shifts,repeat=nd):
                    mask=np.ones_like(expect,dtype=bool);weight=1.
                    for d in range(nd):
                        distance=coords[d]+delta[d]-pos[d]
                        if c['periodic'][d]:distance=(distance+8)%16-8
                        mask &= abs(distance)<2.
                        weight *=kernel[delta[d]]
                    expect += amplitude*weight*mask
                actual=fab['valid'][0]
                err=abs(actual-expect);bound=1e-14+5e-5*abs(expect)
                results.append({'species':name,'component':component,'box':box,'finite_all_fab':bool(np.isfinite(fab['a']).all()),'max_abs_error_K':float(np.max(err)),'max_error_over_bound':float(np.max(err/bound)),'passed':bool(np.isfinite(fab['a']).all() and np.all(err<=bound))})
    return {'oracle':'Exact weighted two-sample variance and cubic positive support, analytically convolved with separable binomial kernel; no moment helper used.','rtol':5e-5,'atol_K':1e-14,'passed':all(x['passed'] for x in results),'max_abs_error_K':max(x['max_abs_error_K'] for x in results),'max_error_over_bound':max(x['max_error_over_bound'] for x in results),'components':results}


if __name__ == "__main__":
    import argparse
    import os
    import subprocess
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (args.output / f"attempt-{attempt:03d}").exists():
        attempt += 1
    run = args.output / f"attempt-{attempt:03d}"
    run.mkdir()
    inputs = args.case.parent / ("inputs_" + case["name"])
    (run / "inputs").write_bytes(inputs.read_bytes())
    command = [str(args.executable.resolve()), "inputs"]
    start = time.monotonic()
    with (run / "stdout.log").open("w") as log:
        result = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT, timeout=75)
    native = None
    for line in (run / "stdout.log").read_text().splitlines():
        if line.startswith(("GHOST_UNITS ", "GHOST_PARTICLES ")):
            native = json.loads(line.split(" ", 1)[1])
    receipt = {
        "case": case, "cwd": str(run), "command": command,
        "registration_ranks": 1, "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "exe_sha256": hashlib.file_digest(args.executable.open("rb"), "sha256").hexdigest(),
        "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "exit_code": result.returncode, "elapsed_s": time.monotonic()-start, "native": native,
    }
    passed = native is not None
    if passed and case["particles"]:
        oracle = particle_oracle(receipt)
        receipt["oracle"] = oracle
        passed = result.returncode == 0 and native["all_fabs_finite"] and oracle["passed"]
    elif passed and case["inject"]:
        passed = result.returncode == 2 and not native["passed"] and not native["all_fabs_finite"] and native["max_valid_error_K"] == 0 and native["max_supported_ghost_error_K"] == 0
    elif passed:
        passed = result.returncode == 0 and native["passed"] and native["all_fabs_finite"]
    receipt["passed"] = bool(passed)
    (run / "RESULT.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"passed": bool(passed), "case": case["name"], "receipt": str(run/"RESULT.json")}))
    raise SystemExit(0 if passed else 1)
