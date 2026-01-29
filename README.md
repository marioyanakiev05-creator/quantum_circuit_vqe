\#quantum\_circuit\_vqe



Simulates quantum circuits using statevector formalism and applies the Variational Quantum Eigensolver (VQE) algorithm to estimate the ground-state energy of the H₂ molecule.

Modular structure ("statevector\_sim.py", "hamiltonian.py", "circuit.py", "vqe.py")



\##Features

* Statevector-based simulation of n-qubit quantum circuits
* Single-qubit and two-qubit gates (X, H, Rx, Ry, Rz, CNOT)
* Measurement and probabilistic sampling
* Quantum noise models:depolarizig, dephasing, amplitude damping
* Implementation of Variational Quantum Eigensolver (VQE)
* Visualization of VQE convergence



\##Tech 



Python, NumPy, Matplotlib  

Run from CLI or in Jupyter Notebook



\##Demo



Current example:  

Ground-state energy estimation of H₂ molecule  

\- Comparison between ideal vs noisy VQE

\- convergence plots



\##Working principle

* Initialize a quantum statevector
* Build a parameterized ansatz circuit
* Apply gates and optional noise channels
* Measure Pauli terms via basis rotations
* Estimate energy from measurement statistics
* Optimize parameters using a classical optimizer
* Output convergence plots
