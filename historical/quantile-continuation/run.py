"""Frozen exact-backup and sampled tabular-learning continuation."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GAMMA = .9
WEIGHTS = np.array([.4, .6])
STEPS, SEEDS, BATCH, CHECK = 20000, 16, 16, 100


def policy_codes(policy):
    return (policy * np.array([4, 2, 1])).sum(axis=-1)


def quantile_projection(z, nxt, reward):
    k = z.shape[-1]
    policy = z.mean(axis=-1).argmax(axis=-1)
    selected = z[np.arange(3), policy]
    targets = (reward[..., None] + GAMMA * selected[nxt]).reshape(3, 2, 2*k)
    order = np.argsort(targets, axis=-1)
    cdf = np.cumsum(np.repeat(WEIGHTS/k, k)[order], axis=-1)
    tau = (np.arange(k)+.5)/k
    indices = (cdf[..., None, :] < tau[:, None]).sum(axis=-1)
    return np.take_along_axis(np.take_along_axis(targets, order, axis=-1), indices, axis=-1)


def exact_grid(nxt, reward, regret):
    summaries, traces = [], {}
    for k in (3, 8, 16, 32, 64):
        for alpha in (1., .5, .1):
            z = np.zeros((3, 2, k)); history = []
            for t in range(4000):
                z += alpha * (quantile_projection(z, nxt, reward)-z)
                if t >= 4000-512: history.append(z.copy())
            history = np.stack(history)
            means = history.mean(axis=-1)
            codes = policy_codes(means.argmax(axis=-1))
            gap = float(np.abs(means[..., 0]-means[..., 1]).min())
            period = None; residual = None
            for p in range(1, 129):
                error = float(np.abs(history[p:]-history[:-p]).max())
                if error < 1e-8: period, residual = p, error; break
            summaries.append({'k':k, 'alpha':alpha, 'period':period, 'residual':residual,
                'strict_policy_cycle':bool(period and period>1 and len(np.unique(codes[-period:]))>1 and gap>1e-4),
                'min_action_gap':gap, 'tail_policy_codes':np.unique(codes).tolist(),
                'tail_mean_regret':float(regret[codes].mean()), 'last_policy_code':int(codes[-1])})
            traces[f'k{k}_a{alpha}'] = history
    return summaries, traces


def gradient_sanity():
    targets = np.array([-2.3, .7, 3.1]); z = .13; tau = .7
    errors = {}
    for loss in ('pinball', 'huber'):
        def objective(value):
            delta = targets-value
            base = np.abs(delta) if loss=='pinball' else np.where(np.abs(delta)<=1, .5*delta**2, np.abs(delta)-.5)
            return np.mean(np.abs(tau-(delta<0))*base)
        delta = targets-z
        increment = np.mean(tau-(delta<0)) if loss=='pinball' else np.mean(np.abs(tau-(delta<0))*np.clip(delta,-1,1))
        derivative = (objective(z+1e-6)-objective(z-1e-6))/2e-6
        errors[loss] = float(abs(derivative+increment))
        assert errors[loss] < 1e-7
    return errors


def schedule(name, t):
    return .05 if name=='constant' else .2/(1+t/1000)**.6


def scalar_learn(nxt, reward, outcomes, name):
    q = np.zeros((SEEDS,3,2)); records=[]
    si = np.arange(SEEDS)[:,None,None,None]
    s = np.arange(3)[None,:,None,None]; a=np.arange(2)[None,None,:,None]
    for t in range(STEPS):
        o = outcomes[t]; ns=nxt[s,a,o]; r=reward[s,a,o]
        targets=r+GAMMA*q.max(axis=-1)[si,ns]
        q += schedule(name,t)*(targets.mean(axis=-1)-q)
        if (t+1)%CHECK==0: records.append(policy_codes(q.argmax(axis=-1)))
    return np.stack(records,axis=1)


def quantile_learn(nxt, reward, outcomes, atom_uniforms, k, loss, name):
    z=np.zeros((SEEDS,3,2,k)); tau=((np.arange(k)+.5)/k)[:,None]
    si=np.arange(SEEDS)[:,None,None,None]
    s=np.arange(3)[None,:,None,None]; a=np.arange(2)[None,None,:,None]
    records=[]
    for t in range(STEPS):
        o=outcomes[t]; ns=nxt[s,a,o]; r=reward[s,a,o]
        policy=z.mean(axis=-1).argmax(axis=-1)
        target_atoms=(atom_uniforms[t]*k).astype(np.int64)
        targets=r+GAMMA*z[si,ns,policy[si,ns],target_atoms]
        delta=targets[...,None,:]-z[...,None]
        if loss=='pinball':
            update=(tau-(delta<0)).mean(axis=-1)
        else:
            update=(np.abs(tau-(delta<0))*np.clip(delta,-1,1)).mean(axis=-1)
        z += schedule(name,t)*update
        if (t+1)%CHECK==0: records.append(policy_codes(z.mean(axis=-1).argmax(axis=-1)))
    assert np.isfinite(z).all()
    return np.stack(records,axis=1), z


def endpoint(codes, baseline, regret, bootstrap_indices):
    tail=codes[:, -40:]; other=baseline[:, -40:]
    per_seed=regret[tail].mean(axis=1); baseline_seed=regret[other].mean(axis=1)
    differences=per_seed-baseline_seed
    interval=np.quantile(differences[bootstrap_indices].mean(axis=1),[.025,.975])
    return {'per_seed_tail_regret':per_seed.tolist(), 'scalar_per_seed_tail_regret':baseline_seed.tolist(),
        'mean_tail_regret':float(per_seed.mean()), 'mean_excess_regret':float(differences.mean()),
        'paired_bootstrap_interval':interval.tolist(), 'tail_suboptimal_fraction':float((regret[tail]>1e-9).mean()),
        'tail_checkpoint_switches_per_seed':np.sum(tail[:,1:]!=tail[:,:-1],axis=1).tolist(),
        'final_policy_codes':codes[:,-1].tolist()}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    args=parser.parse_args(); output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    source=ROOT.parent/'quantile-cycle-result.json'
    witness=json.loads(source.read_text())['witness']
    nxt=np.array(witness['next_states']); reward=np.array(witness['rewards'])
    values=[]
    for code in range(8):
        policy=[(code>>2)&1,(code>>1)&1,code&1]; p=np.zeros((3,3)); r=np.zeros(3)
        for s,a in enumerate(policy):
            for o,w in enumerate(WEIGHTS): p[s,nxt[s,a,o]]+=w; r[s]+=w*reward[s,a,o]
        values.append(np.linalg.solve(np.eye(3)-GAMMA*p,r))
    values=np.stack(values); regret=values[:,0].max()-values[:,0]
    checks=gradient_sanity()
    exact, exact_traces=exact_grid(nxt,reward,regret)
    np.savez_compressed(output/'exact_traces.npz',**exact_traces)
    (output/'exact.json').write_text(json.dumps(exact,indent=2)+'\n')
    print('Exact grid complete',flush=True)
    outcomes=np.empty((STEPS,SEEDS,3,2,BATCH),dtype=np.uint8)
    atom_uniforms=np.empty(outcomes.shape,dtype=np.float32)
    for index,seed in enumerate(range(7100,7116)):
        outcomes[:,index]=(np.random.default_rng(np.random.SeedSequence([seed,0])).random((STEPS,3,2,BATCH))>=.4)
        atom_uniforms[:,index]=np.random.default_rng(np.random.SeedSequence([seed,1])).random((STEPS,3,2,BATCH),dtype=np.float32)
    frequency=float((outcomes==0).mean()); assert abs(frequency-.4)<6*np.sqrt(.4*.6/outcomes.size)
    checks['observed_outcome_zero_frequency']=frequency
    scalar={name:scalar_learn(nxt,reward,outcomes,name) for name in ('constant','decaying')}
    bootstrap_indices=np.random.default_rng(912).integers(SEEDS,size=(10000,SEEDS))
    results=[]; traces={f'scalar_{name}':codes for name,codes in scalar.items()}
    for k in (3,16,32):
        for loss in ('pinball','huber'):
            for name in ('constant','decaying'):
                codes,z=quantile_learn(nxt,reward,outcomes,atom_uniforms,k,loss,name)
                row={'k':k,'loss':loss,'schedule':name,**endpoint(codes,scalar[name],regret,bootstrap_indices)}
                results.append(row); traces[f'k{k}_{loss}_{name}']=codes
                np.save(output/f'final_k{k}_{loss}_{name}.npy',z)
                print(json.dumps({key:row[key] for key in ('k','loss','schedule','mean_excess_regret','paired_bootstrap_interval')}),flush=True)
    np.savez_compressed(output/'sampled_policy_traces.npz',**traces)
    (output/'sampled.json').write_text(json.dumps(results,indent=2)+'\n')
    passing=[]
    for k in (16,32):
        for loss in ('pinball','huber'):
            rows=[r for r in results if r['k']==k and r['loss']==loss]
            if all(r['mean_excess_regret']>=.25 and r['paired_bootstrap_interval'][0]>0 for r in rows): passing.append({'k':k,'loss':loss})
    manifest={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'protocol.md',source)},
        'outcome_bank_sha256':hashlib.sha256(outcomes.tobytes()).hexdigest(),
        'target_uniform_bank_sha256':hashlib.sha256(atom_uniforms.tobytes()).hexdigest(),
        'unique_sampled_transitions':outcomes.size,'quantile_training_transition_consumptions':outcomes.size*12,
        'scalar_training_transition_consumptions':outcomes.size*2,'seed_labels':list(range(7100,7116)),
        'policy_code_values':values.tolist(),'checks':checks,'practical_gate_passing_cells':passing,
        'wall_seconds':time.monotonic()-started}
    manifest['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'practical_gate_passing_cells':passing,'wall_seconds':manifest['wall_seconds']}),flush=True)

if __name__=='__main__': main()
