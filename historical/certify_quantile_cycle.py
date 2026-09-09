"""Certify a discovered orbit using rational affine equations and exact backups."""
from fractions import Fraction as F
from pathlib import Path
import json

root = Path(__file__).resolve().parent
data = json.loads((root / 'quantile-cycle-result.json').read_text())['witness']
next_states, rewards = data['next_states'], data['rewards']
gamma, fractions = F(9,10), (F(1,6), F(1,2), F(5,6))
size = 18

def backup(z):
    means = [sum(z[i:i+3])/3 for i in range(0,size,3)]
    policy = [int(means[2*s+1] > means[2*s]) for s in range(3)]
    gaps = [abs(means[2*s+1]-means[2*s]) for s in range(3)]
    values, sources, offsets = [], [], []
    for s in range(3):
        for a in range(2):
            targets = []
            for o, weight in enumerate((F(2,5),F(3,5))):
                ns = next_states[s][a][o]
                for atom in range(3):
                    index = (ns*2+policy[ns])*3+atom
                    r = F(rewards[s][a][o])
                    targets.append((r+gamma*z[index], weight/3, index, r))
            targets.sort()
            for tau in fractions:
                cumulative = F(0)
                for value, weight, index, r in targets:
                    cumulative += weight
                    if cumulative >= tau:
                        assert cumulative != tau, 'Nonunique quantile boundary'
                        values.append(value); sources.append(index); offsets.append(r)
                        break
    return values, sources, offsets, policy, gaps

approx = [[F(str(x)) for state in phase for action in state for x in action]
          for phase in data['quantile_cycle']]
maps = [backup(z)[1:3] for z in approx]
route, offset = list(range(size)), [F(0)]*size
for sources, constants in maps:
    route, offset = [route[j] for j in sources], [r+gamma*offset[j] for r,j in zip(constants,sources)]
# Solve z = gamma**period * z[route] + offset by exact Gaussian elimination.
matrix = [[F(int(i==j)) for j in range(size)]+[offset[i]] for i in range(size)]
for i,j in enumerate(route): matrix[i][j] -= gamma**len(maps)
for col in range(size):
    pivot = next(row for row in range(col,size) if matrix[row][col])
    matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
    scale = matrix[col][col]
    matrix[col] = [x/scale for x in matrix[col]]
    for row in range(size):
        if row != col:
            scale = matrix[row][col]
            matrix[row] = [x-scale*y for x,y in zip(matrix[row],matrix[col])]
z = [row[-1] for row in matrix]
initial = z.copy()
exact_phases, policies, gaps = [], [], []
for expected_sources, expected_offsets in maps:
    exact_phases.append(z)
    z, sources, offsets, policy, gap = backup(z)
    assert sources == expected_sources and offsets == expected_offsets
    assert min(gap)>0
    policies.append(policy); gaps.extend(gap)
assert z == initial
assert len({tuple(phase) for phase in exact_phases}) == len(maps)
assert len({tuple(policy) for policy in policies}) > 1
optimal_values = [F(3,5),F(0),F(0)]
bad_values = [F(-5025,1309),F(-6725,1309),F(-12105,2618)]
for values, policy in [(optimal_values,[0,0,1]),(bad_values,[0,1,1])]:
    for s, a in enumerate(policy):
        backed_up = sum(weight*(rewards[s][a][o]+gamma*values[next_states[s][a][o]])
                        for o,weight in enumerate((F(2,5),F(3,5))))
        assert values[s] == backed_up
optimal_gaps = []
for s in range(3):
    action_values = [sum(weight*(rewards[s][a][o]+gamma*optimal_values[next_states[s][a][o]])
                         for o,weight in enumerate((F(2,5),F(3,5)))) for a in range(2)]
    assert max(action_values) == optimal_values[s]
    optimal_gaps.append(abs(action_values[0]-action_values[1]))
assert min(optimal_gaps)>0
# Exact initial points of a local attractor; not a claim of global attraction.
result = {'exact_period':len(maps),'exact_closure':True,'policies':policies,
          'minimum_mean_action_gap':str(min(gaps)),
          'minimum_mean_action_gap_float':float(min(gaps)),
          'local_sup_norm_radius':str(min(gaps)/4),
          'period_contraction_factor':str(gamma**len(maps)),
          'quantiles_unique':True,
          'optimal_values':[str(x) for x in optimal_values],
          'bad_policy_values':[str(x) for x in bad_values],
          'optimal_action_gaps':[str(x) for x in optimal_gaps],
          'state_zero_bad_policy_loss':str(optimal_values[0]-bad_values[0]),
          'exact_cycle':[[str(x) for x in phase] for phase in exact_phases]}
print(json.dumps(result,indent=2))
