import subprocess
import sys

def run_lean_code():
    try:
        result = subprocess.run(['lean', 'templates/math/Main.lean'], capture_output=True, text=True, check=True)
        print("Lean output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running Lean code:", e)
        print("Lean error output:")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: Lean executable not found. Make sure Lean is installed and in your PATH.")

if __name__ == "__main__":
    run_lean_code()
