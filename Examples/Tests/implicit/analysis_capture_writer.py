import argparse,json,os,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--output-parent',type=Path,required=True);p.add_argument('--launcher-json',default='[]');a=p.parse_args()
a.output_parent.mkdir(parents=True,exist_ok=True);out=Path(tempfile.mkdtemp(prefix='attempt-',dir=a.output_parent));capture=out/'capture'
cmd=[*json.loads(a.launcher_json),a.executable,'amrex.the_arena_init_size=0','amrex.the_arena_is_managed=0'];env={**os.environ,'WARPX_CONDUCTION_CAPTURE_DIR':str(capture),'OMP_NUM_THREADS':'1'}
(out/'COMMAND.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'stdout.log').open('x') as f:q=subprocess.run(cmd,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT)
assert q.returncode==0,q.returncode
text=(out/'stdout.log').read_text();assert 'REJECTION_OBSERVER_PRE_ROLLBACK_PASS' in text and 'CAPTURE_WRITER_BOUNDED_PASS' in text
records=sorted(capture.glob('event*/CAPTURE.json'));assert len(records)==8
for i,p in enumerate(records):
 d=json.loads(p.read_text());assert d['event']==i+1 and d['terminal']==(i==7) and d['restart_checkpoint'] is False
 assert len(d['fields'])==1 and d['fields'][0]['name']=='state' and (p.parent/'state_H').is_file()
assert not (out/'capture-other').exists()
(out/'PASS.json').write_text(json.dumps({'passed':True,'events':8,'native_exact_rollback_and_off_on':True,'scope':'Generic integrator and bounded native writer; full physical capture qualification remains separate'},indent=2)+'\n')
print('CAPTURE_NATIVE_WRITER_PASS',out,flush=True)
