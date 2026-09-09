"""Exact arithmetic checks for the open-neighborhood robustness theorem."""
from fractions import Fraction as F
from pathlib import Path
import json

root=Path(__file__).resolve().parent
certificate=json.loads((root.parent/'quantile-cycle-certificate.json').read_text())
phases=[[F(x) for x in phase] for phase in certificate['exact_cycle']]
gap=F(certificate['minimum_mean_action_gap'])
radius=gap/4; gamma=F(9,10)
separation=min(max(abs(x-y) for x,y in zip(phases[i],phases[j])) for i in range(6) for j in range(i))
assert separation>2*radius
margins=[]; slopes=[]
for j in range(4):
    for l in range(4):
        mass=F(2*j+3*l,15)
        for tau in (F(1,6),F(1,2),F(5,6)):
            margins.append(abs(mass-tau)); slopes.append(abs(F(j-l,3)))
assert min(margins)==F(1,30) and max(slopes)==1
# An explicit open box inside both the cycle and suboptimality conditions.
delta,eta=F(1,10000),F(1,2000)
h=F(19,250)
assert eta<F(1,30)
assert delta<(1-gamma)*radius
assert delta+5*eta<h*(1-gamma)/2
result={'fixed':'transition destinations, gamma=9/10, K=3, midpoint levels, uniform atom weights',
    'minimum_phase_sup_separation':str(separation),'phase_radius':str(radius),
    'cdf_margin':str(min(margins)),'max_probability_sensitivity':str(max(slopes)),
    'operator_probability_invariance_radius_strict':str(F(1,30)),
    'reward_cycle_bound_strict':str((1-gamma)*radius),
    'explicit_open_box':{'reward_sup_perturbation_lt':str(delta),'per_action_outcome_probability_perturbation_lt':str(eta)},
    'orbit_displacement_bound':str(delta/(1-gamma)),
    'perturbed_true_optimal_gap_lower_bound':str(h-2*(delta+5*eta)/(1-gamma)),
    'all_exact_inequalities_verified':True}
print(json.dumps(result,indent=2))
