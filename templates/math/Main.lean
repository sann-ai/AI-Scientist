-- Basic Lean theorem without using Mathlib
theorem add_comm (a b : Nat) : a + b = b + a := by
  induction a with
  | zero => 
    simp
    rfl
  | succ n ih => 
    simp [Nat.succ_add, ih]
    rfl

#eval "Hello from Lean!"

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

#eval factorial 5
