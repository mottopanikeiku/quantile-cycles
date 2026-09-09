"""Finite exact projected-control search; no sampled RL scores."""
import json
import numpy as np

rng = np.random.default_rng(418)
batch, states, actions, atoms = 512, 3, 2, 3
nxt = rng.integers(states, size=(batch, states, actions, 2))
reward = rng.integers(-3, 4, size=nxt.shape)
z = np.zeros((batch, states, actions, atoms))
rows = np.arange(batch)[:, None, None, None]
trace = []
weights = np.array([.4, .6])
atom_weights = np.repeat(weights / atoms, atoms)
fractions = (np.arange(atoms) + .5) / atoms
for iteration in range(600):
    greedy = z.mean(axis=-1).argmax(axis=-1)
    selected = z[np.arange(batch)[:, None], np.arange(states)[None, :], greedy]
    continuation = selected[rows, nxt]
    targets = (reward[..., None] + .9 * continuation).reshape(batch, states, actions, 6)
    order = np.argsort(targets, axis=-1)
    cumulative = np.cumsum(atom_weights[order], axis=-1)
    indices = (cumulative[..., None, :] < fractions[:, None]).sum(axis=-1)
    z = np.take_along_axis(np.take_along_axis(targets, order, axis=-1), indices, axis=-1)
    if iteration >= 536:
        trace.append(z.copy())
trace = np.stack(trace)
means = trace.mean(axis=-1)
policies = means.argmax(axis=-1)
gaps = np.abs(means[..., 0] - means[..., 1])
witness = None
for world in range(batch):
    for period in range(2, 9):
        residual = float(np.max(np.abs(trace[period:, world] - trace[:-period, world])))
        varying = any(not np.array_equal(policies[-1, world], p) for p in policies[-period:, world])
        if residual < 1e-8 and varying and gaps[-period:, world].min() > 1e-4:
            witness = (world, period, residual)
            break
    if witness is not None:
        break
result = {'seed': 418, 'worlds': batch, 'iterations': 600, 'quantiles': atoms, 'gamma': .9, 'outcome_weights': weights.tolist(), 'witness': None}
if witness is not None:
    world, period, residual = witness
    q = np.zeros((states, actions))
    for _ in range(600):
        q = ((reward[world] + .9*q.max(axis=-1)[nxt[world]]) * weights).sum(axis=-1)
    values = []
    for policy in policies[-period:, world]:
        p = np.zeros((states, states))
        r = np.zeros(states)
        for state, action in enumerate(policy):
            for outcome in range(2):
                p[state, nxt[world, state, action, outcome]] += weights[outcome]
                r[state] += weights[outcome]*reward[world, state, action, outcome]
        values.append(np.linalg.solve(np.eye(states)-.9*p, r).tolist())
    result['witness'] = {'world': world, 'period': period, 'periodicity_residual': residual,
        'next_states': nxt[world].tolist(), 'rewards': reward[world].tolist(),
        'quantile_cycle': trace[-period:, world].tolist(), 'policies': policies[-period:, world].tolist(),
        'min_action_gap': float(gaps[-period:, world].min()), 'true_policy_values': values,
        'optimal_policy': q.argmax(axis=-1).tolist(), 'optimal_values': q.max(axis=-1).tolist()}
print(json.dumps(result, indent=2))
