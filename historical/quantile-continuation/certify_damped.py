"""Exact rational certificate for the observed alpha=1/2 period-26 orbit."""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
w=json.loads((ROOT.parent/'quantile-cycle-result.json').read_text())['witness']
nxt, rewards=w['next_states'],w['rewards']
gamma, alpha, size=F(9,10),F(1,2),18
period=26
history=np.load(ROOT/'results-v1/exact_traces.npz')['k3_a0.5'][-period:]

def projected(z):
    means=[sum(z[i:i+3])/3 for i in range(0,size,3)]
    policy=[int(means[2*s+1]>means[2*s]) for s in range(3)]
    values=[]; routes=[]; offsets=[]
    for s in range(3):
        for a in range(2):
            candidates=[]
            for o,weight in enumerate((F(2,5),F(3,5))):
                ns=nxt[s][a][o]
                for atom in range(3):
                    index=(ns*2+policy[ns])*3+atom
                    candidates.append((F(rewards[s][a][o])+gamma*z[index],weight/3,index,F(rewards[s][a][o])))
            candidates.sort()
            for tau in (F(1,6),F(1,2),F(5,6)):
                mass=F(0)
                for value,weight,index,reward in candidates:
                    mass+=weight
                    if mass>=tau:
                        assert mass!=tau
                        values.append(value); routes.append(index); offsets.append(reward)
                        break
    return values,routes,offsets,policy,[abs(means[2*s+1]-means[2*s]) for s in range(3)]

maps=[projected([F(str(x)) for x in phase.ravel()])[1:3] for phase in history]
# Compose sparse affine maps z'=(1-alpha)z+alpha*(r+gamma*z[route]).
A=[[F(i==j) for j in range(size)] for i in range(size)]
b=[F(0)]*size
for routes,offsets in maps:
    A,b=[[(1-alpha)*A[i][j]+alpha*gamma*A[routes[i]][j] for j in range(size)] for i in range(size)],[(1-alpha)*b[i]+alpha*(offsets[i]+gamma*b[routes[i]]) for i in range(size)]
M=[[F(i==j)-A[i][j] for j in range(size)]+[b[i]] for i in range(size)]
for col in range(size):
    pivot=next(row for row in range(col,size) if M[row][col])
    M[col],M[pivot]=M[pivot],M[col]
    divisor=M[col][col]; M[col]=[v/divisor for v in M[col]]
    for row in range(size):
        if row!=col:
            scale=M[row][col]; M[row]=[x-scale*y for x,y in zip(M[row],M[col])]
z=[row[-1] for row in M]; initial=z.copy()
phases=[]; policies=[]; gaps=[]
for expected_routes,expected_offsets in maps:
    phases.append(z)
    pz,routes,offsets,policy,gap=projected(z)
    assert routes==expected_routes and offsets==expected_offsets
    assert min(gap)>0
    policies.append(policy); gaps.extend(gap)
    z=[(1-alpha)*x+alpha*y for x,y in zip(z,pz)]
assert z==initial
assert len({tuple(x) for x in phases})==period
assert len({tuple(x) for x in policies})>1
result={'alpha':str(alpha),'period':period,'exact_closure':True,
    'minimum_greedy_gap':str(min(gaps)),'minimum_greedy_gap_float':float(min(gaps)),
    'policies':policies,'one_step_local_contraction':str(1-alpha+alpha*gamma),
    'exact_initial_phase':[str(x) for x in initial],
    'minimum_phase_sup_separation':str(min(max(abs(a-b) for a,b in zip(phases[i],phases[j])) for i in range(period) for j in range(i)))}
print(json.dumps(result,indent=2))
