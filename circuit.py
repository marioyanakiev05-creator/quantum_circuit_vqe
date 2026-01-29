import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
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
        gate = self.sim.Rx(theta)
        self.sim.apply_gate(gate, (qubit,))
        self._record('rx', theta=theta, qubit=qubit)

    def ry(self, theta: float, qubit: int):
        gate = self.sim.Ry(theta)
        self.sim.apply_gate(gate, (qubit,))
        self._record('ry', theta=theta, qubit=qubit)

    def rz(self, theta: float, qubit: int):
        gate = self.sim.Rz(theta)
        self.sim.apply_gate(gate, (qubit,))
        self._record('rz', theta=theta, qubit=qubit)

    def s(self, qubit: int):
        self.sim.apply_gate(self.sim.S, (qubit,))
        self._record('s', qubit=qubit)

    def t(self, qubit: int):
        self.sim.apply_gate(self.sim.T, (qubit,))
        self._record('t', qubit=qubit)

    def cnot(self, control: int, target: int):
        self.sim.cnot(control=control, target=target)
        self._record('cnot', control=control, target=target)

    def apply_noise(self, channel: str = 'depolarizing', p: float = 0.01):
        self.sim.apply_noise(channel=channel, p=p)
        self._record('noise', channel=channel, p=p)

    def measure_all(self, shots: int = 1) -> List[int]:
        return self.sim.measure_all(shots=shots)

    def run(self):
        self.sim = StatevectorSimulator(n_qubits=self.n_qubits)
        for op_type, params in self.operations:
            if op_type in ['x', 'z', 'h', 's', 't']:
                qubit = params['qubit']
                gate = getattr(self.sim, op_type.upper())
                self.sim.apply_gate(gate, (qubit,))
            elif op_type in ['rx', 'ry', 'rz']:
                theta = params['theta']
                qubit = params['qubit']
                method_name = op_type[0].upper() + op_type[1:]
                gate_method = getattr(self.sim, method_name)
                gate = gate_method(theta)
                self.sim.apply_gate(gate, (qubit,))
            elif op_type == 'cnot':
                self.sim.cnot(**params)
            elif op_type == 'noise':
                self.sim.apply_noise(**params)
            else:
                raise ValueError(f"Unknown operation: {op_type}")

    def h2_ansatz_ideal(self, theta: np.ndarray):
        if len(theta) != 4:
            raise ValueError("H2 ansatz expects 4 parameters")
        self.sim = StatevectorSimulator(n_qubits=self.n_qubits)
        θ1, θ2, θ3, θ4 = theta
        self.rx(θ1, 0)
        self.ry(θ2, 1)
        self.cnot(0, 1)
        self.rx(θ3, 0)
        self.ry(θ4, 1)

    def h2_ansatz_noisy(self, theta: np.ndarray, noise_profile: List[Dict[str, any]] = None):
        if len(theta) != 4:
            raise ValueError("H2 ansatz expects 4 parameters")
        if noise_profile is None:
            noise_profile = []
        self.sim = StatevectorSimulator(n_qubits=self.n_qubits)
        θ1, θ2, θ3, θ4 = theta
        def apply_noises():
            for noise in noise_profile:
                self.apply_noise(channel=noise['channel'], p=noise['p'])
        self.rx(θ1, 0)
        apply_noises()
        self.ry(θ2, 1)
        apply_noises()
        self.cnot(0, 1)
        apply_noises()
        self.rx(θ3, 0)
        apply_noises()
        self.ry(θ4, 1)
        apply_noises()