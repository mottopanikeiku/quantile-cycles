"""Same exact quantile backups, controlled by a separate scalar Bellman critic."""
from pathlib import Path
import json
import numpy as np

root=Path(__file__).resolve().parent
w=json.loads((root/'quantile-cycle-result.json').read_text())['witness']
nxt=np.array(w['next_states']); r=np.array(w['rewards'])
weights=np.array([.4,.6]); atom_weights=np.repeat(weights/3,3)
fractions=(np.arange(3)+.5)/3
q=np.zeros((3,2)); z=np.zeros((3,2,3)); last=[]
for iteration in range(600):
    policy=q.argmax(axis=-1)
    selected=z[np.arange(3),policy]
    targets=(r[...,None]+.9*selected[nxt]).reshape(3,2,6)
    order=np.argsort(targets,axis=-1)
    cdf=np.cumsum(atom_weights[order],axis=-1)
    indices=(cdf[...,None,:]<fractions[:,None]).sum(axis=-1)
    next_z=np.take_along_axis(np.take_along_axis(targets,order,axis=-1),indices,axis=-1)
    next_q=((r+.9*q.max(axis=-1)[nxt])*weights).sum(axis=-1)
    residual=float(np.max(np.abs(next_q-q)))
    distribution_residual=float(np.max(np.abs(next_z-z)))
    q,z=next_q,next_z
    if iteration>=536:last.append(q.argmax(axis=-1).tolist())
assert all(policy==w['optimal_policy'] for policy in last)
assert residual<1e-10 and distribution_residual<1e-10
print(json.dumps({'comparator':'known scalar-control / distributional-evaluation separation',
    'iterations':600,'tail_policy_switches':sum(a!=b for a,b in zip(last,last[1:])),
    'policy':q.argmax(axis=-1).tolist(),'values':q.max(axis=-1).tolist(),
    'scalar_residual':residual,'distribution_residual':distribution_residual,
    'extra_scalar_entries':6},indent=2))
