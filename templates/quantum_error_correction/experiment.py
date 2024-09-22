import os
import json
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.providers.aer import AerProvider
from qiskit.compiler import transpile
import matplotlib.pyplot as plt

def create_rotated_surface_code(d):
    """
    Create a d x d rotated surface code
    d: distance (odd number)
    """
    n_data = d**2
    n_measure_z = ((d-1)**2 + 1) // 2 + (d-1)  # 内部Z + 外部Z
    n_measure_x = ((d-1)**2 - 1) // 2 + (d-1)  # 内部X + 外部X
    
    q_data = QuantumRegister(n_data, 'data')
    q_measure_z = QuantumRegister(n_measure_z, 'measure_z')
    q_measure_x = QuantumRegister(n_measure_x, 'measure_x')
    c = ClassicalRegister(n_measure_z + n_measure_x, 'c')
    
    qc = QuantumCircuit(q_data, q_measure_z, q_measure_x, c)
    
    # 論理|0>状態の準備（すべてのデータ量子ビットを|0>に初期化）
    for qubit in q_data:
        qc.reset(qubit)
    
    return qc, q_data, q_measure_z, q_measure_x

def apply_error(qc, q_data, error_rate):
    """Apply errors"""
    for qubit in q_data:
        if np.random.random() < error_rate:
            qc.x(qubit)  # Bit flip error
        if np.random.random() < error_rate:
            qc.z(qubit)  # Phase flip error

def measure_syndrome(qc, q_data, q_measure_z, q_measure_x, d):
    # Z症候群の測定
    z_index = 0
    
    # 内部Z症候群
    for row in range(d-1):
        for col in range(d-1):
            if (row + col) % 2 == 0:  # チェッカーボードパターン
                ancilla = q_measure_z[z_index]
                z_index += 1
                qc.cx(q_data[row*d + col], ancilla)
                qc.cx(q_data[row*d + col + 1], ancilla)
                qc.cx(q_data[(row+1)*d + col], ancilla)
                qc.cx(q_data[(row+1)*d + col + 1], ancilla)
    
    # 外部Z症候群（左右の辺）
    for row in range(d-1):
        if row % 2 == 1:  # 左の辺
            ancilla = q_measure_z[z_index]
            z_index += 1
            qc.cx(q_data[row*d], ancilla)
            qc.cx(q_data[(row+1)*d], ancilla)
        
        if row % 2 == 0:  # 右の辺
            ancilla = q_measure_z[z_index]
            z_index += 1
            qc.cx(q_data[row*d + d - 1], ancilla)
            qc.cx(q_data[(row+1)*d + d - 1], ancilla)

    # X症候群の測定
    x_index = 0
    
    # 内部X症候群
    for row in range(d-1):
        for col in range(d-1):
            if (row + col) % 2 == 1:  # チェッカーボードパターン
                ancilla = q_measure_x[x_index]
                x_index += 1
                qc.h(ancilla)
                qc.cx(ancilla, q_data[row*d + col])
                qc.cx(ancilla, q_data[row*d + col + 1])
                qc.cx(ancilla, q_data[(row+1)*d + col])
                qc.cx(ancilla, q_data[(row+1)*d + col + 1])
                qc.h(ancilla)
    
    # 外部X症候群（上下の辺）
    for col in range(d-1):
        # 上の辺
        ancilla = q_measure_x[x_index]
        x_index += 1
        qc.h(ancilla)
        qc.cx(ancilla, q_data[col])
        qc.cx(ancilla, q_data[col + d])
        qc.h(ancilla)
        
        # 下の辺
        ancilla = q_measure_x[x_index]
        x_index += 1
        qc.h(ancilla)
        qc.cx(ancilla, q_data[(d-1)*d + col])
        qc.cx(ancilla, q_data[(d-1)*d + col + d])
        qc.h(ancilla)

    # 測定
    qc.measure(q_measure_z, range(len(q_measure_z)))
    qc.measure(q_measure_x, range(len(q_measure_z), len(q_measure_z) + len(q_measure_x)))

def run_experiment(d, error_rate, shots):
    print(f"d: {d}")
    print(f"q_data length: {d*d}")
    q_data = QuantumRegister(d*d, 'data')
    q_measure_z = QuantumRegister(((d-1)**2 + 1) // 2 + (d-1), 'measure_z')
    q_measure_x = QuantumRegister(((d-1)**2 - 1) // 2 + (d-1), 'measure_x')
    print(f"q_measure_z length: {len(q_measure_z)}")
    print(f"q_measure_x length: {len(q_measure_x)}")
    
    qc, q_data, q_measure_z, q_measure_x = create_rotated_surface_code(d)
    
    # Prepare logical |0> state (initialize all data qubits to |0>)
    
    # Apply errors
    apply_error(qc, q_data, error_rate)
    
    # Syndrome measurement
    measure_syndrome(qc, q_data, q_measure_z, q_measure_x, d)
    
    # Change simulator settings
    provider = AerProvider()
    simulator = provider.get_backend('aer_simulator_statevector_gpu')
    
    # Run simulation
    compiled_circuit = transpile(qc, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    
    # Analyze results
    print(f"Length of measurement results: {len(next(iter(counts)))}")
    print("Z syndrome measurement results:")
    for i, bit in enumerate(next(iter(counts))[:len(q_measure_z)]):
        print(f"Z{i}: {bit}")
    print("X syndrome measurement results:")
    for i, bit in enumerate(next(iter(counts))[len(q_measure_z):]):
        print(f"X{i}: {bit}")

    error_detected = 0
    simulation_results = []  # Use simulation_results instead of results

    for _ in range(shots):
        qc, q_data, q_measure_z, q_measure_x = create_rotated_surface_code(d)
        apply_error(qc, q_data, error_rate)
        measure_syndrome(qc, q_data, q_measure_z, q_measure_x, d)
        
        # Change simulator settings
        simulator = provider.get_backend('aer_simulator_statevector_gpu')
        
        # Run simulation
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts(compiled_circuit)
        
        if next(iter(counts)) != '0' * (len(q_measure_z) + len(q_measure_x)):
            error_detected += 1
        simulation_results.append(result)

    error_detection_rate = error_detected / shots
    return error_detection_rate

def plot_error_correction(d, shots, error_rates, out_dir):
    detection_rates = []
    for rate in error_rates:
        detection_rate = run_experiment(d, rate, shots)
        detection_rates.append(detection_rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates, detection_rates, 'bo-')
    plt.xlabel('Error Rate')
    plt.ylabel('Error Detection Rate')
    plt.title(f'Error Detection Rate for Rotated Surface Code (d={d})')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'error_detection_d{d}.png'))
    plt.close()
    
    return detection_rates

def train(dataset="rotated_surface_code", out_dir="run_0", seed_offset=0):
    d = 3  # Distance (odd number)
    shots = 1000  # Number of executions
    error_rates = np.linspace(0.01, 0.5, 5)  # Range of error rates
    
    detection_rates = plot_error_correction(d, shots, error_rates, out_dir)
    
    results = {
        "d": d,
        "shots": shots,
        "error_rates": error_rates.tolist(),
        "detection_rates": detection_rates
    }

    with open(os.path.join(out_dir, f"results_{dataset}_{seed_offset}.json"), "w") as f:
        json.dump(results, f)

    return results, [], []  # train_log_info と val_log_info は空のリストとして返す

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run quantum error correction experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    num_seeds = {
        "rotated_surface_code": 1,
    }

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    final_infos = {}
    for dataset in num_seeds.keys():
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            final_info, train_info, val_info = train(dataset, out_dir, seed_offset)
            all_results[f"{dataset}_{seed_offset}_final_info"] = final_info
            all_results[f"{dataset}_{seed_offset}_train_info"] = train_info
            all_results[f"{dataset}_{seed_offset}_val_info"] = val_info
            final_info_list.append(final_info)
        final_info_dict = {
            k: [d[k] for d in final_info_list] for k in final_info_list[0].keys()
        }
        means = {f"{k}_mean": np.mean(v) for k, v in final_info_dict.items() if isinstance(v[0], (int, float))}
        stderrs = {
            f"{k}_stderr": np.std(v) / np.sqrt(len(v)) for k, v in final_info_dict.items() if isinstance(v[0], (int, float))
        }
        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict,
        }

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

    print(f"Experiment completed. Results are saved in the {out_dir} directory.")