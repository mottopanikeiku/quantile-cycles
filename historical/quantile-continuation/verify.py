"""Verify bound inputs, output hashes, and semantic deterministic reproduction."""
import hashlib
import json
from pathlib import Path
from fractions import Fraction as F
import numpy as np

root=Path(__file__).resolve().parent
first,second=root/'results-v1',root/'reproduction-v1'
counts={'hashed_files':0,'array_members_compared':0,'json_files_compared':0}
for directory in (first,second):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        path=root.parent/name if name=='quantile-cycle-result.json' else root/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, str(path)
    for name,expected in manifest['output_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==expected, name
        counts['hashed_files']+=1
for path in sorted(first.iterdir()):
    other=second/path.name
    if path.suffix=='.json':
        a=json.loads(path.read_text()); b=json.loads(other.read_text())
        if path.name=='manifest.json':
            for key in ('wall_seconds','output_sha256'): a.pop(key); b.pop(key)
        assert a==b,path.name
        counts['json_files_compared']+=1
    elif path.suffix=='.npz':
        with np.load(path) as a,np.load(other) as b:
            assert a.files==b.files
            for key in a.files:
                assert np.array_equal(a[key],b[key]),(path.name,key)
                counts['array_members_compared']+=1
    elif path.suffix=='.npy':
        assert np.array_equal(np.load(path),np.load(other)),path.name
        counts['array_members_compared']+=1
# Independently solve the eight policy Bellman systems with rational arithmetic.
w=json.loads((root.parent/'quantile-cycle-result.json').read_text())['witness']
exact_values=[]
for code in range(8):
    policy=[(code>>2)&1,(code>>1)&1,code&1]
    matrix=[[F(i==j) for j in range(3)]+[F(0)] for i in range(3)]
    for state,action in enumerate(policy):
        for outcome,weight in enumerate((F(2,5),F(3,5))):
            matrix[state][w['next_states'][state][action][outcome]]-=F(9,10)*weight
            matrix[state][3]+=weight*w['rewards'][state][action][outcome]
    for col in range(3):
        pivot=next(row for row in range(col,3) if matrix[row][col])
        matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        scale=matrix[col][col]; matrix[col]=[x/scale for x in matrix[col]]
        for row in range(3):
            if row!=col:
                scale=matrix[row][col]; matrix[row]=[x-scale*y for x,y in zip(matrix[row],matrix[col])]
    exact_values.append([row[3] for row in matrix])
assert np.allclose(np.array(exact_values,dtype=float),manifest['policy_code_values'],atol=1e-12,rtol=0)
regrets=np.array([float(F(3,5)-v[0]) for v in exact_values])
assert min(regrets)==0
rows=json.loads((first/'sampled.json').read_text())
with np.load(first/'sampled_policy_traces.npz') as traces:
    for row in rows:
        key=f"k{row['k']}_{row['loss']}_{row['schedule']}"
        codes=traces[key][:,-40:]
        baseline=traces[f"scalar_{row['schedule']}"][:,-40:]
        per_seed=regrets[codes].mean(axis=1)
        baseline_seed=regrets[baseline].mean(axis=1)
        assert np.allclose(per_seed,row['per_seed_tail_regret'],atol=1e-12,rtol=0)
        assert np.allclose(baseline_seed,row['scalar_per_seed_tail_regret'],atol=1e-12,rtol=0)
        assert abs(float((per_seed-baseline_seed).mean())-row['mean_excess_regret'])<1e-12
counts['exact_policy_values_verified']=8
counts['sampled_endpoints_recomputed']=len(rows)
print(json.dumps({'status':'verified','semantic_reproduction_identical':True,**counts},indent=2))
