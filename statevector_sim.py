import numpy as np
from typing import Optional, Tuple, List

class StatevectorSimulator:
    """
    Statevector simulator for n qubits.
    """
    
    def __init__(self, n_qubits: int, initial_state: Optional[np.ndarray] = None):
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        
        if initial_state is None:
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[0] = 1.0 + 0j
        else:
            if initial_state.shape != (self.dim,):
                raise ValueError(f"Initial state must have length {self.dim}")
            norm = np.linalg.norm(initial_state)
            if abs(norm - 1.0) > 1e-10:
                raise ValueError("Initial state must be normalized")
            self.state = initial_state.astype(complex)

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

    # Single-qubit gates
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([[cos, -1j * sin], [-1j * sin, cos]], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([[cos, -sin], [sin, cos]], dtype=complex)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        phase = np.exp(-1j * theta / 2)
        return np.array([[phase, 0], [0, np.conj(phase)]], dtype=complex)

    S = Rz(np.pi / 2)
    T = Rz(np.pi / 4)

    def apply_gate(self, gate_matrix: np.ndarray, qubits: Tuple[int, ...]):
        target = qubits[0]

        if not (0 <= target < self.n_qubits):
            raise ValueError(f"Qubit {target} out of range")
        
        full_U = self._build_full_unitary(gate_matrix, target)
        self.state = full_U @ self.state
        self.normalize()

    def _build_full_unitary(self, gate: np.ndarray, target_qubit: int) -> np.ndarray:
        n = self.n_qubits
        ops = [np.eye(2, dtype=complex) for _ in range(n)]
        ops[target_qubit] = gate
        U = ops[0]
        for op in ops[1:]:
            U = np.kron(U, op)
        return U

    def swap_qubits(self, q1: int, q2: int):
        if q1 == q2:
            return
        if q1 > q2:
            q1, q2 = q2, q1
        mask1 = 1 << q1
        mask2 = 1 << q2
        for k in range(self.dim):
            bit1 = (k & mask1) != 0
            bit2 = (k & mask2) != 0
            if bit1 != bit2:
                k_swapped = k ^ mask1 ^ mask2
                if k < k_swapped:
                    temp = self.state[k]
                    self.state[k] = self.state[k_swapped]
                    self.state[k_swapped] = temp

    def cnot(self, control: int, target: int):
        if not (0 <= control < self.n_qubits and 0 <= target < self.n_qubits):
            raise ValueError("Control or target out of range")
        if control == target:
            raise ValueError("Control and target must differ")

        cnot_2q = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

        self.swap_qubits(control, 0)
        self.swap_qubits(target, 1)

        U = cnot_2q
        for _ in range(2, self.n_qubits):
            U = np.kron(U, np.eye(2))

        self.state = U @ self.state
        self.normalize()

    def apply_noise(self, channel: str = 'depolarizing', p: float = 0.01):
        if p < 0 or p > 1:
            raise ValueError("p must be in [0, 1]")

        if channel == 'depolarizing':
            self._apply_depolarizing(p)
        elif channel == 'dephasing':
            self._apply_dephasing(p)
        elif channel == 'amplitude_damping':
            self._apply_amplitude_damping(p)
        else:
            raise ValueError("Unknown channel")

        self.normalize()

    def _apply_depolarizing(self, p: float):
        sqrt_1m3p4 = np.sqrt(1 - 3*p/4)
        sqrt_p4 = np.sqrt(p/4)
        K0 = sqrt_1m3p4 * np.eye(2, dtype=complex)
        K1 = sqrt_p4 * self.X
        K2 = sqrt_p4 * (1j * self.Z @ self.X)
        K3 = sqrt_p4 * self.Z
        K_list = [K0, K1, K2, K3]

        for q in range(self.n_qubits):
            probs = []
            new_states = []
            for K in K_list:
                full_K = self._build_full_unitary(K, q)
                K_psi = full_K @ self.state
                norm_sq = np.sum(np.abs(K_psi)**2)
                probs.append(norm_sq)
                new_states.append(K_psi)

            total = sum(probs)
            if total > 1e-12:
                probs = [pr / total for pr in probs]
            else:
                probs = [1.0] + [0.0] * (len(probs)-1)

            m = np.random.choice(len(probs), p=probs)
            chosen = new_states[m]
            self.state = chosen / np.sqrt(probs[m]) if probs[m] > 0 else chosen
            self.normalize()  # extra safety

    def _apply_dephasing(self, p: float):
        K0 = np.sqrt(1-p) * np.eye(2, dtype=complex)
        K1 = np.sqrt(p) * self.Z
        K_list = [K0, K1]

        for q in range(self.n_qubits):
            probs = []
            new_states = []
            for K in K_list:
                full_K = self._build_full_unitary(K, q)
                K_psi = full_K @ self.state
                norm_sq = np.sum(np.abs(K_psi)**2)
                probs.append(norm_sq)
                new_states.append(K_psi)

            total = sum(probs)
            if total > 1e-12:
                probs = [pr / total for pr in probs]
            else:
                probs = [1.0] + [0.0] * (len(probs)-1)

            m = np.random.choice(len(probs), p=probs)
            chosen = new_states[m]
            self.state = chosen / np.sqrt(probs[m]) if probs[m] > 0 else chosen
            self.normalize()

    def _apply_amplitude_damping(self, p: float):
        K0 = np.array([[1, 0], [0, np.sqrt(1-p)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
        K_list = [K0, K1]

        for q in range(self.n_qubits):
            probs = []
            new_states = []
            for K in K_list:
                full_K = self._build_full_unitary(K, q)
                K_psi = full_K @ self.state
                norm_sq = np.sum(np.abs(K_psi)**2)
                probs.append(norm_sq)
                new_states.append(K_psi)

            total = sum(probs)
            if total > 1e-12:
                probs = [pr / total for pr in probs]
            else:
                probs = [1.0] + [0.0] * (len(probs)-1)

            m = np.random.choice(len(probs), p=probs)
            chosen = new_states[m]
            self.state = chosen / np.sqrt(probs[m]) if probs[m] > 0 else chosen
            self.normalize()

    def get_probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def measure_all(self, shots: int = 1) -> List[int]:
        probs = self.get_probabilities()
        total = np.sum(probs)
        if abs(total - 1.0) > 1e-10:
            probs = probs / total if total > 0 else np.ones_like(probs) / len(probs)
        outcomes = np.random.choice(self.dim, size=shots, p=probs)
        return outcomes.tolist()


