"""Verify the exact post-transient formula from zero, without searching starts."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json

from certify import backup, family, make_model, projection, selected

root=Path(__file__).resolve().parent
rows=[]
for k in list(range(2,17))+[32,64,128]:
    beta=F(1,2*k); f=family(k,beta); model=make_model(f,1)
    # At zero, the tied actions have identical continuation laws. The first
    # Bellman targets are exactly the reward laws, independent of tie-breaking.
    q=[[projection([(f['c'],F(1))],k)[0],projection(list(zip(f['rewards'],f['weights'])),k)[0]]]
    m=F(k+1,2*k)
    m_b=sum(f['qb'][1])/k
    for iteration in range(1,7):
        if iteration>1:
            q,_=backup(q,model,beta,k)
        assert selected(q[0])[1]==(iteration%2)
        if iteration>=2:
            phase=f['qa'] if iteration%2==0 else f['qb']
            shift=beta**(iteration-1)*(m-m_b)
            assert q==[[[x+shift for x in action] for action in phase]]
    rows.append({'k':k,'iterations_checked':6,'exact_translation_formula':True,
                 'shift_at_iteration_two':str(beta*(m-m_b))})
print(json.dumps({'checks':rows,'scope':'Finite checks of the symbolic translation-equivariance induction, not a global-start convergence claim.',
                  'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'certify.py')}},indent=2))
