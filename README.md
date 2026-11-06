
# quantum_circuit_vqe

Simulates quantum circuits using statevector formalism and applies the Variational Quantum Eigensolver (VQE) algorithm to estimate the ground-state energy of the H₂ molecule.

Modular structure (`statevector_sim.py`, `hamiltonian.py`, `circuit.py`, `vqe.py`).

## Features

- Statevector-based simulation of n-qubit quantum circuits
- Single-qubit and two-qubit gates (X, H, Rx, Ry, Rz, CNOT)
- Measurement and probabilistic sampling
- Quantum noise models: depolarizing, dephasing, amplitude damping
- Implementation of Variational Quantum Eigensolver (VQE)
- Exact and measurement-based energy estimation
- Density-matrix simulation for noisy VQE
- Visualization of VQE convergence
- Automated tests for gates, measurements, noise models and VQE

## Tech

Python, NumPy, SciPy, Matplotlib

Run from CLI or in Jupyter Notebook.

## Demo

Current example:

Ground-state energy estimation of the H₂ molecule using a two-qubit Hamiltonian at a bond distance of 0.735 Å.

- Comparison between ideal and noisy VQE
- Comparison with exact Hamiltonian diagonalization
- Convergence plots

Example results:

- Exact ground-state energy: -1.137306 Ha
- Ideal VQE: -1.137306 Ha
- Noisy VQE: approximately -1.087006 Ha (default noise settings)

## Working principle

- Initialize a quantum statevector
- Build a parameterized ansatz circuit using Ry rotations and CNOT
- Apply gates and optional noise channels
- Calculate Hamiltonian expectation values from Pauli terms
- Optimize parameters using a classical optimizer
- Compare optimized energies with exact diagonalization
- Evaluate the effects of noise and finite-shot measurements
- Output convergence plots

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the VQE simulation:

```bash
python vqe.py
```

Run the automated tests:

```bash
python -m unittest discover -s tests -v
```

## Notes

The H₂ example uses a fixed two-qubit Hamiltonian, including nuclear repulsion. Ideal VQE optimization is performed using exact expectation values, while sampled measurements can be used to estimate energies with finite-shot uncertainty.

The noisy simulation uses Kraus operators and density-matrix propagation. The noise parameters are configurable and are not intended to represent a specific quantum processor.

The project uses a consistent qubit-ordering convention and has been tested against analytical gate operations and exact Hamiltonian diagonalization.
