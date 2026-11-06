import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Any

from circuit import Circuit
from hamiltonian import Hamiltonian


def vqe_cost(theta: np.ndarray, circ: Circuit, ham: Hamiltonian,
             noise_profile: Optional[List[Dict[str, Any]]] = None,
             shots: Optional[int] = None, n_averages: int = 1) -> float:
    if n_averages < 1:
        raise ValueError("n_averages must be positive")
    energies = []
    for _ in range(n_averages):
        if noise_profile is None:
            circ.h2_ansatz_ideal(theta)
            energy = (ham.compute_energy_exact(circ.sim) if shots is None
                      else ham.compute_energy(circ.sim, shots))
        else:
            circ.h2_ansatz_noisy(theta, noise_profile)
            energy = ham.compute_energy_density(circ.density_matrix(), shots)
        energies.append(energy)
    return float(np.mean(energies))


def run_vqe(initial_theta: np.ndarray, ham: Hamiltonian,
            noise_profile: Optional[List[Dict[str, Any]]] = None,
            shots: Optional[int] = None, maxiter: int = 300,
            n_runs: int = 1) -> Tuple[float, np.ndarray, List[float]]:
    if len(initial_theta) != 4:
        raise ValueError("H2 ansatz expects 4 initial parameters")
    if n_runs < 1:
        raise ValueError("n_runs must be positive")

    circ = Circuit(n_qubits=2)
    rng = np.random.default_rng(2026)
    best = None
    best_history = []

    for run in range(n_runs):
        theta0 = initial_theta if run == 0 else rng.uniform(-np.pi, np.pi, 4)
        history = [vqe_cost(theta0, circ, ham, noise_profile)]

        def objective(theta):
            return vqe_cost(theta, circ, ham, noise_profile)

        def callback(theta):
            history.append(objective(theta))

        result = minimize(objective, theta0, method='BFGS',
                          callback=callback, options={'maxiter': maxiter, 'gtol': 1e-7})
        history.append(objective(result.x))
        if best is None or result.fun < best.fun:
            best, best_history = result, history
        print(f"Run {run + 1}: {result.fun:.8f} Ha ({result.message})")

    if shots is not None:
        estimate = vqe_cost(best.x, circ, ham, noise_profile,
                            shots=shots, n_averages=5)
        print(f"Final sampled estimate ({shots} shots/term): {estimate:.8f} Ha")

    return float(best.fun), best.x, best_history


def plot_convergence(history_ideal: List[float], history_noisy: List[float],
                     exact_energy: Optional[float] = None):
    plt.figure(figsize=(10, 6))
    plt.plot(history_ideal, label='Ideal', linewidth=2)
    plt.plot(history_noisy, label='Noisy', linewidth=2)
    if exact_energy is not None:
        plt.axhline(exact_energy, linestyle='--', color='gray', label='Exact')
    plt.xlabel('Iteration')
    plt.ylabel('Total energy (Ha)')
    plt.title('VQE: ideal and noisy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ham = Hamiltonian()
    initial_theta = np.array([0.2, -0.7, 0.3, 0.4])

    print('Running ideal VQE...')
    energy_ideal, theta_ideal, history_ideal = run_vqe(
        initial_theta, ham, shots=5000, n_runs=2)
    print(f'Ideal optimized energy: {energy_ideal:.8f} Ha')

    noise_config = [
        {'channel': 'depolarizing', 'p': 0.01},
        {'channel': 'dephasing', 'p': 0.005},
    ]
    print('\nRunning noisy VQE...')
    energy_noisy, theta_noisy, history_noisy = run_vqe(
        initial_theta, ham, noise_config, shots=5000, n_runs=2)
    print(f'Noisy optimized energy: {energy_noisy:.8f} Ha')

    exact_e0 = ham.exact_ground_energy()
    print(f'\nExact H2 energy: {exact_e0:.8f} Ha')
    print(f'Ideal error: {abs(energy_ideal - exact_e0):.8f} Ha')
    print(f'Noisy error: {abs(energy_noisy - exact_e0):.8f} Ha')
    plot_convergence(history_ideal, history_noisy, exact_e0)
