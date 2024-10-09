import subprocess
import sys

def run_lean_code():
    try:
        result = subprocess.run(['lean', 'templates/math/Main.lean'], capture_output=True, text=True, check=False)
        print("Lean output:")
        print(result.stdout)
        if result.returncode != 0:
            print("Error running Lean code. Exit code:", result.returncode)
            print("Lean error output:")
            print(result.stderr)
    except FileNotFoundError:
        print("Error: Lean executable not found. Make sure Lean is installed and in your PATH.")

if __name__ == "__main__":
    run_lean_code()
