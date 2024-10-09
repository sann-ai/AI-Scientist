import Mathlib.Tactic
import Mathlib.Data.Real.Basic

theorem example_theorem : ∀ (x y : ℝ), x + y = y + x := by
  intros x y
  exact add_comm x y

#eval "Hello from Lean!"

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

#eval factorial 5
