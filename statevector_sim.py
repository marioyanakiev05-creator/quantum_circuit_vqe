import numpy as np
from typing import Optional, Tuple, List


class StatevectorSimulator:
    """Statevector simulator using |q0 q1 ... q(n-1)> ordering."""

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        phase = np.exp(-1j * theta / 2)
        return np.diag([phase, np.conj(phase)])

    S = Rz(np.pi / 2)
    T = Rz(np.pi / 4)

    def __init__(self, n_qubits: int, initial_state: Optional[np.ndarray] = None):
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits

        if initial_state is None:
            self.state = np.zeros(self.dim, dtype=complex)
            self.state[0] = 1.0
        else:
            state = np.asarray(initial_state, dtype=complex)
            if state.shape != (self.dim,):
                raise ValueError(f"Initial state must have length {self.dim}")
            if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-10):
                raise ValueError("Initial state must be normalized")
            self.state = state.copy()

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm == 0:
            raise ValueError("Cannot normalize a zero state")
        self.state /= norm

    def _mask(self, qubit: int) -> int:
        if not (0 <= qubit < self.n_qubits):
            raise ValueError(f"Qubit {qubit} out of range")
        return 1 << (self.n_qubits - 1 - qubit)

    def apply_gate(self, gate_matrix: np.ndarray, qubits: Tuple[int, ...]):
        if len(qubits) != 1:
            raise ValueError("apply_gate expects one target qubit")
        gate = np.asarray(gate_matrix, dtype=complex)
        if gate.shape != (2, 2):
            raise ValueError("Single-qubit gate must have shape (2, 2)")
        self.state = self._build_full_unitary(gate, qubits[0]) @ self.state

    def _build_full_unitary(self, gate: np.ndarray, target_qubit: int) -> np.ndarray:
        self._mask(target_qubit)
        ops = [np.eye(2, dtype=complex) for _ in range(self.n_qubits)]
        ops[target_qubit] = gate
        full = ops[0]
        for op in ops[1:]:
            full = np.kron(full, op)
        return full

    def swap_qubits(self, q1: int, q2: int):
        mask1, mask2 = self._mask(q1), self._mask(q2)
        if q1 == q2:
            return
        for i in range(self.dim):
            if bool(i & mask1) != bool(i & mask2):
                j = i ^ mask1 ^ mask2
                if i < j:
                    self.state[i], self.state[j] = self.state[j], self.state[i]

    def _cnot_unitary(self, control: int, target: int) -> np.ndarray:
        control_mask, target_mask = self._mask(control), self._mask(target)
        if control == target:
            raise ValueError("Control and target must differ")
        full = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.dim):
            j = i ^ target_mask if (i & control_mask) else i
            full[j, i] = 1.0
        return full

    def cnot(self, control: int, target: int):
        control_mask, target_mask = self._mask(control), self._mask(target)
        if control == target:
            raise ValueError("Control and target must differ")
        for i in range(self.dim):
            if (i & control_mask) and not (i & target_mask):
                j = i | target_mask
                self.state[i], self.state[j] = self.state[j], self.state[i]

    @classmethod
    def kraus_operators(cls, channel: str, p: float) -> List[np.ndarray]:
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        if channel == 'depolarizing':
            return [np.sqrt(1 - 3 * p / 4) * np.eye(2),
                    np.sqrt(p / 4) * cls.X,
                    np.sqrt(p / 4) * cls.Y,
                    np.sqrt(p / 4) * cls.Z]
        if channel == 'dephasing':
            return [np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * cls.Z]
        if channel == 'amplitude_damping':
            return [np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex),
                    np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)]
        raise ValueError(f"Unknown channel: {channel}")

    def apply_noise(self, channel: str = 'depolarizing', p: float = 0.01):
        kraus = self.kraus_operators(channel, p)
        for q in range(self.n_qubits):
            states = [self._build_full_unitary(K, q) @ self.state for K in kraus]
            probs = np.array([np.vdot(v, v).real for v in states])
            probs /= probs.sum()
            index = np.random.choice(len(states), p=probs)
            self.state = states[index] / np.linalg.norm(states[index])

    def get_probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def measure_all(self, shots: int = 1) -> List[int]:
        if shots < 1:
            raise ValueError("shots must be positive")
        probs = self.get_probabilities()
        probs /= probs.sum()
        return np.random.choice(self.dim, size=shots, p=probs).tolist()
