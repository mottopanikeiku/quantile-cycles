import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith

set_option maxRecDepth 4096
set_option maxHeartbeats 4000000
/-!
The rational K = 2, beta = 1/4 example in THEOREM.md, Section 6.
This file does not formalize the real, arbitrary-K attraction theorem.
Targets are exact reward/continuation product laws, not a lookup by phase.
-/
namespace QuantileCycles

inductive Action where
  | A | B
  deriving DecidableEq, Repr

abbrev Atoms := ℚ × ℚ
abbrev Table := Atoms × Atoms
abbrev Outcome := ℚ × ℚ
abbrev Law := List Outcome

def beta : ℚ := 1 / 4

def rewardLaw : Action → Law
  | .A => [(91 / 128, 1)]
  | .B => [(0, 3 / 16), (1 / 2, 1 / 2), (1, 5 / 16)]

def mass (law : Law) : ℚ := (law.map Prod.snd).sum

def expected (law : Law) : ℚ := (law.map fun o => o.1 * o.2).sum

def ValidLaw (law : Law) : Prop :=
  mass law = 1 ∧ ∀ o ∈ law, 0 < o.2

def BoundedRewards (law : Law) : Prop :=
  ∀ o ∈ law, 0 ≤ o.1 ∧ o.1 ≤ 1

/-- There is exactly one environment state; every action returns to it. -/
def nextState (_ : Unit) (_ : Action) : Unit := ()

def mean (x : Atoms) : ℚ := (x.1 + x.2) / 2

def atoms (q : Table) : Action → Atoms
  | .A => q.1
  | .B => q.2

/-- A is used only to resolve a tie; neither certified phase has a tie. -/
def greedy (q : Table) : Action :=
  if mean q.2 ≤ mean q.1 then .A else .B

/-- Each reward outcome is paired with both equally likely continuation atoms. -/
def targetLaw (law : Law) (x : Atoms) : Law :=
  law.flatMap fun o => [(o.1 + beta * x.1, o.2 / 2),
                       (o.1 + beta * x.2, o.2 / 2)]

def cdf (law : Law) (z : ℚ) : ℚ :=
  (law.map fun o => if o.1 ≤ z then o.2 else 0).sum

def leftCdf (law : Law) (z : ℚ) : ℚ :=
  (law.map fun o => if o.1 < z then o.2 else 0).sum

/-- Least support value whose CDF reaches tau. The maximum support value seeds
    the minimum scan: for a normalized law and tau ≤ 1 it is itself eligible.
    The input seed is a support value, not a fallback for a missing quantile. -/
def inverse (law : Law) (tau : ℚ) : ℚ :=
  let upper := law.foldl (fun z o => max z o.1) (law.head!.1)
  law.foldl (fun z o => if tau ≤ cdf law o.1 then min z o.1 else z) upper

def project (law : Law) : Atoms := (inverse law (1 / 4), inverse law (3 / 4))

/-- The synchronous mean-greedy, hard midpoint-quantile Bellman operator. -/
def F (q : Table) : Table :=
  let continuation := atoms q (greedy q)
  (project (targetLaw (rewardLaw .A) continuation),
   project (targetLaw (rewardLaw .B) continuation))

/-- Generalized inverse over the rational domain, not just a support-index test. -/
def IsInverse (law : Law) (tau z : ℚ) : Prop :=
  tau ≤ cdf law z ∧ ∀ y : ℚ, y < z → cdf law y < tau

/-- A strict CDF jump also excludes the nonunique quantile-boundary case. -/
def StrictJump (law : Law) (tau z : ℚ) : Prop :=
  leftCdf law z < tau ∧ tau < cdf law z

private theorem cdf_le_leftCdf (law : Law) (z y : ℚ)
    (hp : ∀ o ∈ law, 0 ≤ o.2) (hy : y < z) :
    cdf law y ≤ leftCdf law z := by
  induction law with
  | nil => simp [cdf, leftCdf]
  | cons o rest ih =>
    have ho := hp o (by simp)
    have hr : ∀ v ∈ rest, 0 ≤ v.2 := by
      intro v hv
      exact hp v (by simp [hv])
    have hi := ih hr
    simp only [cdf, leftCdf, List.map_cons, List.sum_cons] at hi ⊢
    split_ifs <;> linarith

 theorem strictJump_isInverse (law : Law) (tau z : ℚ)
    (hp : ∀ o ∈ law, 0 ≤ o.2) (h : StrictJump law tau z) :
    IsInverse law tau z := by
  refine ⟨le_of_lt h.2, ?_⟩
  intro y hy
  exact lt_of_le_of_lt (cdf_le_leftCdf law z y hp hy) h.1

 theorem inverse_unique (law : Law) (tau x y : ℚ)
    (hx : IsInverse law tau x) (hy : IsInverse law tau y) : x = y := by
  apply le_antisymm
  · by_contra h
    have := hx.2 y (lt_of_not_ge h)
    linarith [hy.1]
  · by_contra h
    have := hy.2 x (lt_of_not_ge h)
    linarith [hx.1]

/-- The two exact tables, in action order (A,B), each with two quantile atoms. -/
def Q_A : Table := ((107 / 120, 61 / 60), (1307 / 1920, 2267 / 1920))
def Q_B : Table := ((1793 / 1920, 1853 / 1920), (347 / 480, 587 / 480))

 theorem reward_laws_valid : ValidLaw (rewardLaw .A) ∧ ValidLaw (rewardLaw .B) := by
  norm_num [ValidLaw, mass, rewardLaw]

 theorem rewards_bounded : BoundedRewards (rewardLaw .A) ∧ BoundedRewards (rewardLaw .B) := by
  norm_num [BoundedRewards, rewardLaw]

 theorem discount_valid : 0 < beta ∧ beta < 1 := by decide

 theorem expected_rewards : expected (rewardLaw .A) = 91 / 128 ∧
    expected (rewardLaw .B) = 9 / 16 := by decide

 theorem strict_reward_gap : expected (rewardLaw .A) - expected (rewardLaw .B) = 19 / 128 ∧
    expected (rewardLaw .B) < expected (rewardLaw .A) := by decide

/-- Expected-return Bellman backup; it uses the reward law, not critic means. -/
def trueBackup (a : Action) (v : ℚ) : ℚ := expected (rewardLaw a) + beta * v

def V_A : ℚ := 91 / 96

def V_B : ℚ := 3 / 4

 theorem stationary_values : trueBackup .A V_A = V_A ∧ trueBackup .B V_B = V_B ∧
    V_A - V_B = 19 / 96 := by decide

/-- A is strictly better for every common continuation value. -/
 theorem true_action_optimal (v : ℚ) : trueBackup .B v < trueBackup .A v := by
  norm_num [trueBackup, expected, rewardLaw, beta]

/-- Bellman optimality and uniqueness of its fixed point for this MDP. -/
 theorem optimal_bellman (v : ℚ) :
    max (trueBackup .A v) (trueBackup .B v) = v ↔ v = V_A := by
  rw [max_eq_left (le_of_lt (true_action_optimal v))]
  norm_num [trueBackup, expected, rewardLaw, beta, V_A]
  constructor <;> intro h <;> linarith

/-- Stationary randomized policy: p is the probability of taking A. -/
def policyValue (p : ℚ) : ℚ :=
  (p * expected (rewardLaw .A) + (1 - p) * expected (rewardLaw .B)) / (1 - beta)

 theorem randomized_policy_optimal (p : ℚ) (hp : 0 ≤ p) (hp' : p ≤ 1) :
    policyValue p ≤ V_A ∧ (policyValue p = V_A ↔ p = 1) := by
  norm_num [policyValue, expected, rewardLaw, beta, V_A]
  constructor
  · linarith
  · constructor <;> intro h <;> linarith

 theorem strict_greedy_phases :
    mean Q_A.1 - mean Q_A.2 = 3 / 128 ∧
    mean Q_B.2 - mean Q_B.1 = 3 / 128 ∧
    greedy Q_A = .A ∧ greedy Q_B = .B := by decide

 theorem phase_atoms_sorted :
    Q_A.1.1 < Q_A.1.2 ∧ Q_A.2.1 < Q_A.2.2 ∧
    Q_B.1.1 < Q_B.1.2 ∧ Q_B.2.1 < Q_B.2.2 := by decide

/-- These are the actual product target laws in the two backups. -/
 theorem cycle_targets_valid :
    ValidLaw (targetLaw (rewardLaw .A) Q_A.1) ∧
    ValidLaw (targetLaw (rewardLaw .B) Q_A.1) ∧
    ValidLaw (targetLaw (rewardLaw .A) Q_B.2) ∧
    ValidLaw (targetLaw (rewardLaw .B) Q_B.2) := by
  norm_num [ValidLaw, mass, targetLaw, rewardLaw, Q_A, Q_B, beta]

 theorem cycle_quantiles_strict :
    StrictJump (targetLaw (rewardLaw .A) Q_A.1) (1 / 4) Q_B.1.1 ∧
    StrictJump (targetLaw (rewardLaw .A) Q_A.1) (3 / 4) Q_B.1.2 ∧
    StrictJump (targetLaw (rewardLaw .B) Q_A.1) (1 / 4) Q_B.2.1 ∧
    StrictJump (targetLaw (rewardLaw .B) Q_A.1) (3 / 4) Q_B.2.2 ∧
    StrictJump (targetLaw (rewardLaw .A) Q_B.2) (1 / 4) Q_A.1.1 ∧
    StrictJump (targetLaw (rewardLaw .A) Q_B.2) (3 / 4) Q_A.1.2 ∧
    StrictJump (targetLaw (rewardLaw .B) Q_B.2) (1 / 4) Q_A.2.1 ∧
    StrictJump (targetLaw (rewardLaw .B) Q_B.2) (3 / 4) Q_A.2.2 := by decide

/-- Every computed midpoint on either phase satisfies the generalized inverse
    condition for all rational thresholds y below it, not just target atoms. -/
 theorem cycle_quantiles_correct (q : Table) (hq : q = Q_A ∨ q = Q_B)
    (a : Action) (tau : ℚ) (ht : tau = 1 / 4 ∨ tau = 3 / 4) :
    IsInverse (targetLaw (rewardLaw a) (atoms q (greedy q))) tau
      (inverse (targetLaw (rewardLaw a) (atoms q (greedy q))) tau) := by
  rcases hq with rfl | rfl <;> cases a <;> rcases ht with rfl | rfl
  all_goals
    apply strictJump_isInverse
    · norm_num [targetLaw, rewardLaw, atoms, greedy, mean, Q_A, Q_B, beta]
    · decide

 theorem cycle_closure : F Q_A = Q_B ∧ F Q_B = Q_A := by decide

 theorem phases_distinct : Q_A ≠ Q_B := by decide

 theorem strict_suboptimal_two_cycle :
    F Q_A = Q_B ∧ F Q_B = Q_A ∧ Q_A ≠ Q_B ∧
    greedy Q_A = .A ∧ greedy Q_B = .B ∧
    trueBackup (greedy Q_B) V_A < trueBackup .A V_A := by
  exact ⟨cycle_closure.1, cycle_closure.2, phases_distinct,
    strict_greedy_phases.2.2.1, strict_greedy_phases.2.2.2,
    by rw [strict_greedy_phases.2.2.2]; exact true_action_optimal V_A⟩

#print axioms strictJump_isInverse
#print axioms inverse_unique
#print axioms reward_laws_valid
#print axioms rewards_bounded
#print axioms discount_valid
#print axioms expected_rewards
#print axioms strict_reward_gap
#print axioms stationary_values
#print axioms true_action_optimal
#print axioms optimal_bellman
#print axioms randomized_policy_optimal
#print axioms strict_greedy_phases
#print axioms phase_atoms_sorted
#print axioms cycle_targets_valid
#print axioms cycle_quantiles_strict
#print axioms cycle_quantiles_correct
#print axioms cycle_closure
#print axioms phases_distinct
#print axioms strict_suboptimal_two_cycle

end QuantileCycles
