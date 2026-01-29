import numpy as np
from typing import Dict, List, Tuple
from statevector_sim import StatevectorSimulator

class Hamiltonian:
    def __init__(self):
        self.terms: Dict[str, float] = {
            'I':    -0.04207897647782288,
            'Z0':   -0.7888869056042214,
            'Z1':   -0.7888869056042214,
            'Z0Z1':  0.17072816152119294,
            'X0X1':  0.22279687776746033,
            'Y0Y1':  0.22279687776746033,
        }
        self.paulis: List[Tuple[str, float]] = list(self.terms.items())

    def compute_energy(self, sim: StatevectorSimulator, shots: int = 1000) -> float:
        sim.normalize()  # force norm=1 before measurements
        energy = 0.0
        for pauli_str, coef in self.paulis:
            if pauli_str == 'I':
                exp_val = 1.0
            else:
                exp_val = self._measure_pauli(sim, pauli_str, shots=shots)
            energy += coef * exp_val
        return energy

    def _measure_pauli(self, sim: StatevectorSimulator, pauli_str: str, shots: int = 1000) -> float:
        temp_sim = StatevectorSimulator(n_qubits=sim.n_qubits)
        temp_sim.state = sim.state.copy()
        temp_sim.normalize()  # extra safety

        active_qubits = []
        active_paulis = []
        
        i = 0
        while i < len(pauli_str):
            char = pauli_str[i]
            if char in 'XYZI':
                qubit_idx = 0
                i += 1
                while i < len(pauli_str) and pauli_str[i].isdigit():
                    qubit_idx = qubit_idx * 10 + int(pauli_str[i])
                    i += 1
                if not (0 <= qubit_idx < sim.n_qubits):
                    raise ValueError(f"Invalid qubit index {qubit_idx} in '{pauli_str}'")
                active_qubits.append(qubit_idx)
                active_paulis.append(char)
            else:
                i += 1

        if not active_qubits:
            return 1.0

        for idx, pauli_char in enumerate(active_paulis):
            q = active_qubits[idx]
            if pauli_char == 'X':
                temp_sim.apply_gate(temp_sim.H, (q,))
            elif pauli_char == 'Y':
                temp_sim.apply_gate(temp_sim.Rz(-np.pi/2), (q,))
                temp_sim.apply_gate(temp_sim.H, (q,))
            elif pauli_char == 'Z':
                pass
            elif pauli_char == 'I':
                pass
            else:
                raise ValueError(f"Unknown Pauli: {pauli_char}")

        outcomes = temp_sim.measure_all(shots=shots)

        exp_val = 0.0
        for outcome in outcomes:
            bit_count = 0
            for idx, pauli_char in enumerate(active_paulis):
                if pauli_char in ['X', 'Y', 'Z']:
                    q = active_qubits[idx]
                    bit = (outcome >> q) & 1
                    bit_count += bit
            exp_val += 1 if bit_count % 2 == 0 else -1

        return exp_val / shots