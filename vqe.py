import numpy as np
from scipy.optimize import minimize
from circuit import Circuit
from hamiltonian import Hamiltonian
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Any

def vqe_cost(theta: np.ndarray, circ: Circuit, ham: Hamiltonian, noise_profile: Optional[List[Dict[str, Any]]] = None, shots: int = 20000, n_averages: int = 5) -> float:
    energies = []
    for _ in range(n_averages):
        if noise_profile is None:
            circ.h2_ansatz_ideal(theta)
        else:
            circ.h2_ansatz_noisy(theta, noise_profile)
        energy = ham.compute_energy(circ.sim, shots=shots)
        energies.append(energy)
    return np.mean(energies)

def run_vqe(initial_theta: np.ndarray, ham: Hamiltonian, noise_profile: Optional[List[Dict[str, Any]]] = None, shots: int = 20000, maxiter: int = 200, n_runs: int = 1) -> Tuple[float, np.ndarray, List[float]]:
    circ = Circuit(n_qubits=2)
    history = []
    
    def callback(xk):
        current_energy = vqe_cost(xk, circ, ham, noise_profile, shots, n_averages=3)
        history.append(current_energy)
        print(f"Iteration energy: {current_energy:.6f}")

    result = minimize(
        fun=vqe_cost,
        x0=initial_theta,
        args=(circ, ham, noise_profile, shots, 5),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'disp': True},
        callback=callback
    )

    return result.fun, result.x, history

def plot_convergence(history_ideal: List[float], history_noisy: List[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(history_ideal, label='Ideal', linewidth=2)
    plt.plot(history_noisy, label='Noisy', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy (Ha)')
    plt.title('VQE Convergence: Ideal vs Noisy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ham = Hamiltonian()
    initial_theta = np.random.uniform(-np.pi, np.pi, 4)

    print("Running ideal VQE...")
    energy_ideal, theta_ideal, history_ideal = run_vqe(initial_theta, ham, None, shots=5000, maxiter=200)
    print(f"Ideal optimized energy: {energy_ideal:.6f}")
    print(f"Ideal optimized theta: {theta_ideal}")

    noise_config = [
        {'channel': 'depolarizing', 'p': 0.01},
        {'channel': 'dephasing', 'p': 0.005}
    ]
    print("\nRunning noisy VQE...")
    energy_noisy, theta_noisy, history_noisy = run_vqe(initial_theta, ham, noise_config, shots=5000, maxiter=200)
    print(f"Noisy optimized energy: {energy_noisy:.6f}")
    print(f"Noisy optimized theta: {theta_noisy}")

    plot_convergence(history_ideal, history_noisy)

    exact_e0 = -1.136
    print("\nExact ground energy:", exact_e0)
    print("Ideal error:", abs(energy_ideal - exact_e0))
    print("Noisy error:", abs(energy_noisy - exact_e0))