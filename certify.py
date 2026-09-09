"""Exact finite-instance certificates for the analytic arbitrary-K cycle family."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import platform
import time

ROOT=Path(__file__).resolve().parent


def projection(targets, k):
    targets=sorted(targets)
    assert sum(prob for _,prob in targets)==1
    atoms=[]; margin=F(1)
    for i in range(1,k+1):
        tau=F(2*i-1,2*k); cumulative=F(0)
        for value,prob in targets:
            previous=cumulative; cumulative+=prob
            if cumulative>=tau:
                assert previous<tau<cumulative, 'Quantile must be unique'
                atoms.append(value); margin=min(margin,tau-previous,cumulative-tau)
                break
    return atoms,margin


def selected(critic):
    means=[sum(atoms)/len(atoms) for atoms in critic]
    best=max(range(len(means)),key=means.__getitem__)
    assert len(means)==1 or means[0]!=means[1], 'No action tie'
    return critic[best],best


def backup(q, model, discount, k):
    choices=[selected(state)[0] for state in q]
    out=[]; margin=F(1)
    for state in model:
        actions=[]
        for outcomes in state:
            targets=[(reward+discount*x,prob/k) for reward,prob,nxt in outcomes for x in choices[nxt]]
            atoms,action_margin=projection(targets,k)
            actions.append(atoms); margin=min(margin,action_margin)
        out.append(actions)
    return out,margin


def family(k,beta):
    assert k>=2 and 0<beta<=F(1,2*k)
    eps=F(1,4*k*k); m=F(k+1,2*k); spread=F(k-1,2*k)
    c=m-beta*(1+beta)*spread/2
    weights=[F(1,2*k)-eps]+[F(1,k)]*(k-1)+[F(1,2*k)+eps]
    rewards=[F(i,k) for i in range(k+1)]
    assert sum(weights)==1 and min(weights)>0 and 0<c<1
    mean_b=sum(p*r for p,r in zip(weights,rewards))
    assert mean_b==F(1,2)+eps
    lower=F(6*k*k-3*k+1,16*k**3)
    assert c-mean_b>=lower>F(1,4*k)
    m_b=(m+beta*c-beta*beta*spread)/(1-beta*beta)
    m_a=c+beta*m_b
    s=[F(i,k)-m for i in range(1,k+1)]
    qa=[[c+beta*(m_b+x) for x in s],[F(i,k)+beta*(m_b-spread) for i in range(1,k+1)]]
    qb=[[c+beta*(m_a+beta*x) for x in s],[F(i,k)+beta*(m_a-beta*spread) for i in range(1,k+1)]]
    assert selected(qa)[1]==0 and selected(qb)[1]==1
    gap=beta*(1-beta)*spread/2
    for q in (qa,qb):
        assert abs(sum(q[0])/k-sum(q[1])/k)==gap
    assert beta*(max(selected(qa)[0])-min(selected(qa)[0]))<F(1,k)
    assert beta*(max(selected(qb)[0])-min(selected(qb)[0]))<F(1,k)
    return {'qa':qa,'qb':qb,'c':c,'weights':weights,'rewards':rewards,'mean_b':mean_b,'gap':gap,'lower':lower}


def make_model(f,length):
    nxt=1 if length>1 else 0
    model=[[[ (f['c'],F(1),nxt) ],[(r,p,nxt) for r,p in zip(f['rewards'],f['weights'])]]]
    for j in range(1,length):
        model.append([[(F(0),F(1),(j+1)%length)]])
    return model


def policy_value_check(f,model,gamma,length):
    beta=gamma**length
    policy_values=[]
    for action,mean in ((0,f['c']),(1,f['mean_b'])):
        v0=mean/(1-beta)
        values=[v0]+[gamma**(length-j)*v0 for j in range(1,length)]
        for state in range(length):
            outcomes=model[state][action if state==0 else 0]
            assert values[state]==sum(p*(r+gamma*values[ns]) for r,p,ns in outcomes)
        policy_values.append(v0)
    assert policy_values[0]>policy_values[1]
    # A is also strictly greedy at its exact Bellman fixed point.
    assert f['c']+beta*policy_values[0]>f['mean_b']+beta*policy_values[0]
    return policy_values


def one_state(k):
    beta=F(1,2*k); f=family(k,beta); model=make_model(f,1)
    a,ma=backup([f['qa']],model,beta,k)
    b,mb=backup([f['qb']],model,beta,k)
    assert a==[f['qb']] and b==[f['qa']] and a!=b
    values=policy_value_check(f,model,beta,1)
    return {'k':k,'states':1,'discount':str(beta),'exact_period':2,'exact_closure':True,
        'greedy_word':[0,1],'greedy_gap':str(f['gap']),'cdf_margin':str(min(ma,mb)),
        'true_reward_gap':str(f['c']-f['mean_b']),'true_reward_gap_lower_bound':str(f['lower']),
        'frozen_bad_policy_loss':str(values[0]-values[1]),'local_phase_radius':str(f['gap']/4),
        'two_step_contraction':str(beta*beta)}


def delayed(k,gamma):
    length=1
    while gamma**length>F(1,2*k):length+=1
    beta=gamma**length; f=family(k,beta); model=make_model(f,length)
    def phase(t):return 0 if t%(2*length)<length else 1
    decisions=[f['qa'],f['qb']]
    # Forced-state arrays can be shared across phases without mutation.
    forced={j:[[gamma**(length-j)*x for x in selected(d)[0]] for d in decisions] for j in range(1,length)}
    orbit=[]
    for t in range(2*length):
        q=[decisions[phase(t)]]
        q.extend([forced[j][phase(t-(length-j))]] for j in range(1,length))
        orbit.append(q)
    minimum_margin=F(1)
    for t,q in enumerate(orbit):
        advanced,margin=backup(q,model,gamma,k)
        assert advanced==orbit[(t+1)%(2*length)],(k,t)
        assert selected(q[0])[1]==phase(t)
        minimum_margin=min(minimum_margin,margin)
    flattened=[tuple(x for state in q for action in state for x in action) for q in orbit]
    assert len(set(flattened))==2*length
    values=policy_value_check(f,model,gamma,length)
    return {'k':k,'states':length,'discount':str(gamma),'effective_discount':str(beta),
        'exact_period':2*length,'all_primitive_backups_verified':True,'all_phases_distinct':True,
        'greedy_word_runs':[[0,length],[1,length]],'greedy_gap':str(f['gap']),
        'cdf_margin':str(minimum_margin),'true_reward_gap':str(f['c']-f['mean_b']),
        'frozen_bad_policy_loss':str(values[0]-values[1]),
        'local_phase_radius':str(f['gap']/4),'full_period_contraction':str(gamma**(2*length))}


def main():
    started=time.monotonic()
    results={'one_state':[one_state(k) for k in list(range(2,17))+[32,64,128]],
        'fixed_discount':[delayed(k,F(9,10)) for k in (2,3,8,16,32)],
        'scope':'Finite exact checks support the separately proved analytic family; not a practical learning experiment.',
        'python':platform.python_version(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'construction_sha256':hashlib.sha256((ROOT/'construction.md').read_bytes()).hexdigest()}
    results['wall_seconds']=time.monotonic()-started
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
