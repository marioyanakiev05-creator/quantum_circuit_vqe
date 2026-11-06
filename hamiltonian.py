import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from statevector_sim import StatevectorSimulator


class Hamiltonian:
    """Two-qubit H2 Hamiltonian (STO-3G, 0.735 angstrom bond length)."""

    def __init__(self):
        self.terms: Dict[str, float] = {
            'I': -1.052373245772859,
            'Z0': 0.39793742484318045,
            'Z1': -0.39793742484318045,
            'Z0Z1': -0.01128010425623538,
            'X0X1': 0.18093119978423156,
        }
        self.nuclear_repulsion = 0.71996899
        self.paulis: List[Tuple[str, float]] = list(self.terms.items())

    @staticmethod
    def _operators(pauli_str: str, n_qubits: int) -> List[str]:
        ops = ['I'] * n_qubits
        if pauli_str == 'I':
            return ops
        tokens = re.findall(r'[IXYZ]\d+', pauli_str)
        if ''.join(tokens) != pauli_str:
            raise ValueError(f"Invalid Pauli term: {pauli_str}")
        used = set()
        for token in tokens:
            qubit = int(token[1:])
            if not 0 <= qubit < n_qubits or qubit in used:
                raise ValueError(f"Invalid qubit index in: {pauli_str}")
            ops[qubit] = token[0]
            used.add(qubit)
        return ops

    def pauli_matrix(self, pauli_str: str, n_qubits: int = 2) -> np.ndarray:
        matrices = {
            'I': np.eye(2, dtype=complex),
            'X': StatevectorSimulator.X,
            'Y': StatevectorSimulator.Y,
            'Z': StatevectorSimulator.Z,
        }
        ops = self._operators(pauli_str, n_qubits)
        full = matrices[ops[0]]
        for op in ops[1:]:
            full = np.kron(full, matrices[op])
        return full

    def matrix(self) -> np.ndarray:
        return sum(coef * self.pauli_matrix(term) for term, coef in self.paulis)

    def exact_ground_energy(self) -> float:
        return float(np.linalg.eigvalsh(self.matrix())[0] + self.nuclear_repulsion)

    def compute_energy_exact(self, sim: StatevectorSimulator) -> float:
        energy = np.vdot(sim.state, self.matrix() @ sim.state).real
        return float(energy + self.nuclear_repulsion)

    def compute_energy(self, sim: StatevectorSimulator, shots: int = 1000) -> float:
        if shots < 1:
            raise ValueError("shots must be positive")
        energy = self.nuclear_repulsion
        for term, coef in self.paulis:
            expectation = 1.0 if term == 'I' else self._measure_pauli(sim, term, shots)
            energy += coef * expectation
        return float(energy)

    def _measure_pauli(self, sim: StatevectorSimulator,
                       pauli_str: str, shots: int = 1000) -> float:
        ops = self._operators(pauli_str, sim.n_qubits)
        if all(op == 'I' for op in ops):
            return 1.0

        temp = StatevectorSimulator(sim.n_qubits, sim.state)
        for q, op in enumerate(ops):
            if op == 'X':
                temp.apply_gate(temp.H, (q,))
            elif op == 'Y':
                temp.apply_gate(temp.Rz(-np.pi / 2), (q,))
                temp.apply_gate(temp.H, (q,))

        outcomes = np.asarray(temp.measure_all(shots))
        parity = np.zeros(shots, dtype=int)
        for q, op in enumerate(ops):
            if op != 'I':
                parity += (outcomes >> (sim.n_qubits - 1 - q)) & 1
        return float(np.mean(1 - 2 * (parity % 2)))

    def compute_energy_density(self, rho: np.ndarray,
                               shots: Optional[int] = None) -> float:
        if rho.shape != (4, 4):
            raise ValueError("Expected a 4 x 4 density matrix")
        if shots is not None and shots < 1:
            raise ValueError("shots must be positive")
        energy = self.nuclear_repulsion
        for term, coef in self.paulis:
            expectation = np.trace(rho @ self.pauli_matrix(term)).real
            if shots is not None and term != 'I':
                prob_plus = np.clip((1 + expectation) / 2, 0, 1)
                expectation = 2 * np.random.binomial(shots, prob_plus) / shots - 1
            energy += coef * expectation
        return float(energy)
