import json, sys, glob, statistics as st
import numpy as np
def calls(run):
    rows=[]
    for f in glob.glob(f'results/kaggle/{run}/traces/*.json'):
        t=json.load(open(f)); prev=None
        for s in t['steps']:
            ex=s.get('extra',{}); obs=(s.get('observation') or {}).get('extra',{})
            if s.get('source')=='agent' and s.get('metrics') and prev is not None and 'timestamp' in ex:
                m=s['metrics']; dt=ex['timestamp']-prev
                if dt>0: rows.append((m['prompt_tokens'], m['completion_tokens'], dt))
            prev = obs.get('timestamp', ex.get('timestamp', prev))
    return np.array(rows, float)
for run in sys.argv[1:]:
    r=calls(run); X=np.c_[np.ones(len(r)), r[:,0]/1000, r[:,1]/1000]; y=r[:,2]
    coef,*_=np.linalg.lstsq(X,y,rcond=None); pred=X@coef; r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
    print(f'{run}: n={len(r)} median call {np.median(y):.1f}s, median prompt {np.median(r[:,0]):.0f} tok, median out {np.median(r[:,1]):.0f} tok')
    print(f'   fit: {coef[0]:.1f}s + {coef[1]:.2f}s per 1k prompt + {coef[2]:.1f}s per 1k output (R2 {r2:.2f}); decode ~{1000/coef[2]:.0f} tok/s')
    print(f'   share of call time: prompt {coef[1]*r[:,0].sum()/1000/y.sum():.0%}, output {coef[2]*r[:,1].sum()/1000/y.sum():.0%}')
