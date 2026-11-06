import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from statevector_sim import StatevectorSimulator


class Circuit:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.sim = StatevectorSimulator(n_qubits=n_qubits)
        self.operations: List[Tuple[str, dict]] = []

    def _record(self, op_type: str, **kwargs):
        self.operations.append((op_type, kwargs))

    def x(self, qubit: int):
        self.sim.apply_gate(self.sim.X, (qubit,))
        self._record('x', qubit=qubit)

    def z(self, qubit: int):
        self.sim.apply_gate(self.sim.Z, (qubit,))
        self._record('z', qubit=qubit)

    def h(self, qubit: int):
        self.sim.apply_gate(self.sim.H, (qubit,))
        self._record('h', qubit=qubit)

    def rx(self, theta: float, qubit: int):
        self.sim.apply_gate(self.sim.Rx(theta), (qubit,))
        self._record('rx', theta=theta, qubit=qubit)

    def ry(self, theta: float, qubit: int):
        self.sim.apply_gate(self.sim.Ry(theta), (qubit,))
        self._record('ry', theta=theta, qubit=qubit)

    def rz(self, theta: float, qubit: int):
        self.sim.apply_gate(self.sim.Rz(theta), (qubit,))
        self._record('rz', theta=theta, qubit=qubit)

    def s(self, qubit: int):
        self.sim.apply_gate(self.sim.S, (qubit,))
        self._record('s', qubit=qubit)

    def t(self, qubit: int):
        self.sim.apply_gate(self.sim.T, (qubit,))
        self._record('t', qubit=qubit)

    def cnot(self, control: int, target: int):
        self.sim.cnot(control, target)
        self._record('cnot', control=control, target=target)

    def apply_noise(self, channel: str = 'depolarizing', p: float = 0.01):
        self.sim.apply_noise(channel, p)
        self._record('noise', channel=channel, p=p)

    def measure_all(self, shots: int = 1) -> List[int]:
        return self.sim.measure_all(shots)

    def _reset(self):
        self.sim = StatevectorSimulator(self.n_qubits)
        self.operations = []

    def _gate(self, name: str, params: dict) -> np.ndarray:
        if name in ('x', 'z', 'h', 's', 't'):
            return getattr(self.sim, name.upper())
        if name in ('rx', 'ry', 'rz'):
            return getattr(self.sim, name[0].upper() + name[1:])(params['theta'])
        raise ValueError(f"Unknown gate: {name}")

    def run(self):
        self.sim = StatevectorSimulator(self.n_qubits)
        for name, params in self.operations:
            if name == 'cnot':
                self.sim.cnot(**params)
            elif name == 'noise':
                self.sim.apply_noise(**params)
            else:
                self.sim.apply_gate(self._gate(name, params), (params['qubit'],))

    def density_matrix(self) -> np.ndarray:
        rho = np.zeros((self.sim.dim, self.sim.dim), dtype=complex)
        rho[0, 0] = 1.0
        for name, params in self.operations:
            if name == 'noise':
                kraus = self.sim.kraus_operators(**params)
                for q in range(self.n_qubits):
                    updated = np.zeros_like(rho)
                    for K in kraus:
                        full = self.sim._build_full_unitary(K, q)
                        updated += full @ rho @ full.conj().T
                    rho = updated
            else:
                if name == 'cnot':
                    full = self.sim._cnot_unitary(**params)
                else:
                    full = self.sim._build_full_unitary(self._gate(name, params), params['qubit'])
                rho = full @ rho @ full.conj().T
        return rho

    def h2_ansatz_ideal(self, theta: np.ndarray):
        if self.n_qubits != 2 or len(theta) != 4:
            raise ValueError("H2 ansatz expects 2 qubits and 4 parameters")
        self._reset()
        self.ry(theta[0], 0)
        self.ry(theta[1], 1)
        self.cnot(0, 1)
        self.ry(theta[2], 0)
        self.ry(theta[3], 1)

    def h2_ansatz_noisy(self, theta: np.ndarray,
                        noise_profile: Optional[List[Dict[str, Any]]] = None):
        if self.n_qubits != 2 or len(theta) != 4:
            raise ValueError("H2 ansatz expects 2 qubits and 4 parameters")
        self._reset()
        noise_profile = noise_profile or []

        def apply_noises():
            for noise in noise_profile:
                self.apply_noise(channel=noise['channel'], p=noise['p'])

        self.ry(theta[0], 0)
        apply_noises()
        self.ry(theta[1], 1)
        apply_noises()
        self.cnot(0, 1)
        apply_noises()
        self.ry(theta[2], 0)
        apply_noises()
        self.ry(theta[3], 1)
        apply_noises()
